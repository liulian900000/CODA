"""
Patch 与文本匹配的单文件核心实现。

功能：
1. 读取 patch 特征 .pt 文件或目录。
2. 读取 CSV 中的组织/文本描述。
3. 使用 CONCH 编码文本。
4. 计算 patch 与文本的相似度矩阵。
5. 对每个文本保留 top-k patch，并保存为 [文本数, topk, 特征维度]。

输入约定：
- patch .pt 文件默认是形状 [N, D] 的 torch.Tensor。
- CSV 至少两列：第一列为组织/类别名称，第二列为文本描述。

输出 .pt 内容：
{
    "tissue_types": List[str],
    "selected_features": Tensor[K, topk, D],
    "tissue_topk_indices": Tensor[K, topk],
    "tissue_topk_scores": Tensor[K, topk],
}
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

try:
    from CONCH.conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
except ImportError as exc:
    raise ImportError(
        "无法导入 CONCH。请先安装/配置 CONCH 包，确保可以导入 "
        "CONCH.conch.open_clip_custom。"
    ) from exc


logger = logging.getLogger("Patch_Match")


class CONCHTextEncoder:
    """最小化的 CONCH 文本编码器。"""

    def __init__(
        self,
        model_name: str = "conch_ViT-B-16",
        checkpoint_path: str = "hf_hub:MahmoodLab/CONCH",
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("加载 CONCH 模型: %s | checkpoint=%s | device=%s", model_name, checkpoint_path, self.device)

        self.model, _ = create_model_from_pretrained(model_name, checkpoint_path=checkpoint_path)
        self.model = self.model.to(self.device).eval()
        self.tokenizer = get_tokenizer()

    @torch.no_grad()
    def encode_texts(self, descriptions: Sequence[str], add_quotes: bool = True) -> torch.Tensor:
        """编码文本并返回 CPU 上的 L2 归一化特征 [K, D]。"""
        features = []
        for text in tqdm(descriptions, desc="编码文本", leave=False):
            text = str(text)
            if add_quotes:
                text = f'"{text}"'
            tokens = self.tokenizer([text], return_tensors="pt")["input_ids"].to(self.device)
            feat = self.model.encode_text(tokens)
            features.append(feat)

        text_features = torch.cat(features, dim=0)
        text_features = F.normalize(text_features, p=2, dim=-1)
        return text_features.cpu()


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_patch_features(pt_file_path: str, normalize: bool = False) -> torch.Tensor:
    """读取单个 patch 特征文件，返回 float Tensor [N, D]。"""
    path = Path(pt_file_path)
    if not path.exists():
        raise FileNotFoundError(f"patch 文件不存在: {path}")

    data = torch.load(path, map_location="cpu")
    if isinstance(data, dict):
        for key in ("features", "patch_features", "feat", "feats"):
            if key in data:
                data = data[key]
                break

    if not isinstance(data, torch.Tensor):
        raise ValueError(f"{path} 中未找到 Tensor patch 特征，当前类型: {type(data)}")
    if data.ndim != 2:
        raise ValueError(f"patch 特征必须是 2D Tensor [N, D]，当前 shape={tuple(data.shape)}")

    features = data.float()
    if normalize:
        features = F.normalize(features, p=2, dim=-1)
    return features


def load_text_descriptions(csv_path: str) -> Tuple[List[str], List[str]]:
    """读取 CSV 前两列：类别/组织名称、文本描述。"""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV 文件不存在: {path}")

    last_error = None
    df = None
    for encoding in ("utf-8", "utf-8-sig", "gbk", "gb2312"):
        try:
            df = pd.read_csv(path, encoding=encoding)
            break
        except UnicodeDecodeError as exc:
            last_error = exc

    if df is None:
        raise RuntimeError(f"读取 CSV 编码失败: {last_error}")
    if df.shape[1] < 2:
        raise ValueError(f"CSV 至少需要两列，当前列数: {df.shape[1]}")

    tissue_types = df.iloc[:, 0].astype(str).tolist()
    descriptions = df.iloc[:, 1].astype(str).tolist()
    valid_rows = [
        (t.strip(), d.strip())
        for t, d in zip(tissue_types, descriptions)
        if t and d and t.strip() and d.strip() and t.lower() != "nan" and d.lower() != "nan"
    ]
    if not valid_rows:
        raise ValueError("CSV 中没有有效的文本描述")

    tissue_types, descriptions = zip(*valid_rows)
    return list(tissue_types), list(descriptions)


def compute_similarity_matrix(
    patch_features: torch.Tensor,
    text_features: torch.Tensor,
    similarity_type: str = "cosine",
    device: Optional[str] = None,
) -> torch.Tensor:
    """计算 patch-text 相似度矩阵 [N, K]。"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    patch_features = patch_features.to(device)
    text_features = text_features.to(device)

    if similarity_type == "cosine":
        patch_features = F.normalize(patch_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)
    elif similarity_type != "dot":
        raise ValueError("similarity_type 只支持 'cosine' 或 'dot'")

    return patch_features @ text_features.T


def select_topk_patches(
    patch_features: torch.Tensor,
    similarity_matrix: torch.Tensor,
    topk: int = 300,
    pad_to_topk: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    对每个文本分别选 top-k patch。

    Returns:
        selected_features: [K, topk, D] 或 [K, min(topk, N), D]
        topk_indices: [K, topk]
        topk_scores: [K, topk]
    """
    if topk <= 0:
        raise ValueError("topk 必须大于 0")

    patch_features = patch_features.cpu()
    similarity_matrix = similarity_matrix.cpu()
    num_patches, num_texts = similarity_matrix.shape
    actual_k = min(topk, num_patches)

    selected_features_list = []
    topk_indices_list = []
    topk_scores_list = []

    for text_idx in range(num_texts):
        scores = similarity_matrix[:, text_idx]
        top_scores, top_indices = torch.topk(scores, actual_k)
        feats = patch_features[top_indices]

        if pad_to_topk and actual_k < topk:
            pad_n = topk - actual_k
            feats = F.pad(feats, (0, 0, 0, pad_n), mode="constant", value=0)
            top_indices = F.pad(top_indices, (0, pad_n), mode="constant", value=-1)
            top_scores = F.pad(top_scores, (0, pad_n), mode="constant", value=float("nan"))

        selected_features_list.append(feats)
        topk_indices_list.append(top_indices)
        topk_scores_list.append(top_scores)

    selected_features = torch.stack(selected_features_list, dim=0)
    topk_indices = torch.stack(topk_indices_list, dim=0)
    topk_scores = torch.stack(topk_scores_list, dim=0)
    return selected_features, topk_indices, topk_scores


def match_single_slide(
    pt_file_path: str,
    text_features: torch.Tensor,
    tissue_types: Sequence[str],
    output_dir: Optional[str] = None,
    topk: int = 300,
    similarity_type: str = "cosine",
    normalize_patch: bool = False,
    save_similarity: bool = False,
    device: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """处理单个切片：patch 与所有文本匹配，并保留每个文本 top-k patch。"""
    patch_features = load_patch_features(pt_file_path, normalize=normalize_patch)
    similarity_matrix = compute_similarity_matrix(patch_features, text_features, similarity_type, device)
    selected_features, topk_indices, topk_scores = select_topk_patches(
        patch_features=patch_features,
        similarity_matrix=similarity_matrix,
        topk=topk,
        pad_to_topk=True,
    )

    result = {
        "tissue_types": list(tissue_types),
        "selected_features": selected_features,
        "tissue_topk_indices": topk_indices,
        "tissue_topk_scores": topk_scores,
    }
    if save_similarity:
        result["similarity_matrix"] = similarity_matrix.cpu()

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{Path(pt_file_path).stem}.pt"
        torch.save(result, out_path)
        logger.info("保存: %s | selected_features=%s", out_path, tuple(selected_features.shape))

    return result


def run_patch_text_matching(
    patch_path: str,
    csv_file: str,
    output_dir: str,
    file_pattern: str = "*.pt",
    topk: int = 300,
    model_name: str = "conch_ViT-B-16",
    checkpoint_path: str = "hf_hub:MahmoodLab/CONCH",
    similarity_type: str = "cosine",
    normalize_patch: bool = False,
    save_similarity: bool = False,
    device: Optional[str] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """批量或单文件执行 patch-text top-k 匹配。"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tissue_types, descriptions = load_text_descriptions(csv_file)
    logger.info("读取到 %d 条文本描述", len(descriptions))

    encoder = CONCHTextEncoder(model_name=model_name, checkpoint_path=checkpoint_path, device=device)
    text_features = encoder.encode_texts(descriptions)
    logger.info("文本特征 shape=%s", tuple(text_features.shape))

    patch_path_obj = Path(patch_path)
    if patch_path_obj.is_dir():
        pt_files = sorted(patch_path_obj.glob(file_pattern))
    else:
        pt_files = [patch_path_obj]
    if not pt_files:
        raise FileNotFoundError(f"未找到 patch 特征文件: {patch_path} | pattern={file_pattern}")

    all_results = {}
    failed = []
    for pt_file in tqdm(pt_files, desc="匹配切片"):
        try:
            result = match_single_slide(
                pt_file_path=str(pt_file),
                text_features=text_features,
                tissue_types=tissue_types,
                output_dir=output_dir,
                topk=topk,
                similarity_type=similarity_type,
                normalize_patch=normalize_patch,
                save_similarity=save_similarity,
                device=device,
            )
            all_results[pt_file.stem] = result
        except Exception as exc:
            logger.exception("处理失败: %s | %s", pt_file, exc)
            failed.append({"file": str(pt_file), "error": str(exc)})

    summary = {
        "total": len(pt_files),
        "success": len(all_results),
        "failed": failed,
        "tissue_types": tissue_types,
        "topk": topk,
        "similarity_type": similarity_type,
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "patch_match_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("完成: %d/%d 成功", len(all_results), len(pt_files))
    return all_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patch 与文本匹配，并为每个文本保留 top-k patch")
    parser.add_argument("--patch_path", required=True, help="patch 特征 .pt 文件或目录")
    parser.add_argument("--csv_file", required=True, help="文本描述 CSV，前两列分别为名称和描述")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--file_pattern", default="*.pt", help="patch_path 是目录时的文件匹配模式")
    parser.add_argument("--topk", type=int, default=300, help="每个文本保留 top-k patch")
    parser.add_argument("--model_name", default="conch_ViT-B-16", help="CONCH 模型名称")
    parser.add_argument("--checkpoint_path", default="hf_hub:MahmoodLab/CONCH", help="CONCH checkpoint 路径")
    parser.add_argument("--similarity_type", choices=["cosine", "dot"], default="cosine", help="相似度类型")
    parser.add_argument("--normalize_patch", action="store_true", help="读取 patch 后先做 L2 归一化")
    parser.add_argument("--save_similarity", action="store_true", help="是否额外保存完整相似度矩阵 [N, K]")
    parser.add_argument("--device", default=None, help="cuda/cpu，默认自动选择")
    parser.add_argument("--log_level", default="INFO", help="日志级别")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    run_patch_text_matching(
        patch_path=args.patch_path,
        csv_file=args.csv_file,
        output_dir=args.output_dir,
        file_pattern=args.file_pattern,
        topk=args.topk,
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        similarity_type=args.similarity_type,
        normalize_patch=args.normalize_patch,
        save_similarity=args.save_similarity,
        device=args.device,
    )


if __name__ == "__main__":
    main()
