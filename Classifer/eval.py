import argparse

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from data import ConditionManager, StructuredPatchDataset
from models import IntraTissueGatedAttention, TissueDiffusionModel
from train import eval_stage1, eval_stage2_likelihood
from utils import get_alphas_cumprod


def build_loader(dataset, batch_size, indices):
    if len(indices) == 0:
        return None
    subset = torch.utils.data.Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


def main():
    parser = argparse.ArgumentParser(description='Evaluate pretrained CODA classifier checkpoints')
    parser.add_argument('--feature_dir', type=str, required=True, help='预提取特征目录')
    parser.add_argument('--label_csv', type=str, required=True, help='标签CSV')
    parser.add_argument('--normal_text_pt', type=str, required=True, help='正常文本特征')
    parser.add_argument('--abnormal_text_pt', type=str, required=True, help='肿瘤文本特征')
    parser.add_argument('--topk', type=int, default=200, help='每tissue选取的patch数')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--eval_steps', type=int, default=100, help='扩散评估采样步数')
    parser.add_argument('--stage1_ckpt', type=str, required=True, help='Stage 1 聚合器权重')
    parser.add_argument('--stage2_ckpt', type=str, required=True, help='Stage 2 扩散模型权重')
    parser.add_argument('--threshold', type=float, default=None, help='固定阈值；不传则自动在验证集搜索')
    parser.add_argument('--save_stage1_csv', type=str, default='stage1_eval_diagnosis.csv')
    parser.add_argument('--save_stage2_csv', type=str, default='stage2_eval_diagnosis.csv')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    cond_mgr = ConditionManager(args.normal_text_pt, args.abnormal_text_pt, device)
    alphas = get_alphas_cumprod(1000, device)
    target_tissues = cond_mgr.target_tissues

    full_ds = StructuredPatchDataset(
        args.label_csv,
        args.feature_dir,
        target_tissues=target_tissues,
        topk=args.topk,
    )

    test_idxs = [i for i, d in enumerate(full_ds.valid_data) if d['slide_id'].startswith('test')]
    train_val_idxs = [i for i, d in enumerate(full_ds.valid_data) if not d['slide_id'].startswith('test')]

    val_idxs = []
    if len(train_val_idxs) > 0:
        tv_labels = [full_ds.valid_data[i]['label'] for i in train_val_idxs]
        train_idxs, val_idxs = train_test_split(train_val_idxs, test_size=0.2, stratify=tv_labels, random_state=42)

    test_loader = build_loader(full_ds, args.batch_size, test_idxs)
    val_loader = build_loader(full_ds, args.batch_size, val_idxs)

    if test_loader is None:
        raise RuntimeError('No test samples found. Expected slide_id starting with "test".')

    aggregator = IntraTissueGatedAttention(num_tissues=len(target_tissues), topk=args.topk).to(device)
    diffusion = TissueDiffusionModel(num_layers=2).to(device)

    aggregator.load_state_dict(torch.load(args.stage1_ckpt, map_location=device))
    diffusion.load_state_dict(torch.load(args.stage2_ckpt, map_location=device))
    aggregator.eval()
    diffusion.eval()

    train_z = []
    source_loader = val_loader if val_loader is not None else test_loader
    with torch.no_grad():
        for x, mask, _, _ in source_loader:
            train_z.append(aggregator(x.to(device), mask.to(device)).cpu())
    train_z = torch.cat(train_z, dim=0)
    global_mean = train_z.mean(dim=0, keepdim=True).to(device)
    global_std = train_z.std(dim=0, keepdim=True).to(device) + 1e-6

    print('\n>>> EVALUATING STAGE 1 ON TEST SET')
    stage1_metrics = eval_stage1(aggregator, test_loader, device, save_csv=args.save_stage1_csv)
    print(f"Test AUC:         {stage1_metrics['metrics']['auc']:.4f}")
    print(f"Test Acc:         {stage1_metrics['metrics']['acc']:.4f}")
    print(f"Test F1:          {stage1_metrics['metrics']['f1']:.4f}")
    print(f"Test Recall:      {stage1_metrics['metrics']['recall']:.4f}")
    print(f"Test Specificity: {stage1_metrics['metrics']['specificity']:.4f}")

    print('\n>>> EVALUATING STAGE 2 ON TEST SET')
    val_threshold = args.threshold
    if val_threshold is None and val_loader is not None:
        print('No fixed threshold provided, searching threshold on validation split...')
        val_metrics = eval_stage2_likelihood(
            aggregator,
            diffusion,
            val_loader,
            cond_mgr,
            alphas,
            device,
            global_mean,
            global_std,
            steps=args.eval_steps,
            fixed_thresh=None,
            save_csv=None,
        )
        val_threshold = float(val_metrics['metrics']['best_thresh'])
        print(f'Chosen threshold from validation: {val_threshold:.6f}')
    elif val_threshold is None:
        print('No validation split available; using threshold 0.0')
        val_threshold = 0.0

    test_metrics = eval_stage2_likelihood(
        aggregator,
        diffusion,
        test_loader,
        cond_mgr,
        alphas,
        device,
        global_mean,
        global_std,
        steps=args.eval_steps,
        fixed_thresh=val_threshold,
        save_csv=args.save_stage2_csv,
    )
    m = test_metrics['metrics']
    print(f"Test AUC:         {m['auc']:.4f}")
    print(f"Test Acc:         {m['acc']:.4f}")
    print(f"Test F1:          {m['f1']:.4f}")
    print(f"Test Recall:      {m['recall']:.4f}")
    print(f"Test Specificity: {m['specificity']:.4f}")
    print(f"Best Threshold:   {m['best_thresh']:.6f}")


if __name__ == '__main__':
    main()
