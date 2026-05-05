import os
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


LABEL_MAP = {"normal": 0, "tumor": 1}


@dataclass
class SlideSample:
    slide_id: str
    path: str
    label: int


class ConditionManager:
    def __init__(self, normal_pt_path: str, abnormal_pt_path: str, device: torch.device):
        self.device = device
        norm_data = torch.load(normal_pt_path, map_location=device)
        tumor_data = torch.load(abnormal_pt_path, map_location=device)

        self.target_tissues = norm_data["tissue_types"]
        self.norm_feats = torch.nn.functional.normalize(norm_data["text_features"], p=2, dim=1)
        self.tumor_feats = torch.nn.functional.normalize(tumor_data["text_features"], p=2, dim=1)

        print(f"ConditionManager initialized with {len(self.target_tissues)} tissues: {self.target_tissues}")
        print("\n=== [ConditionManager] Condition Difference Analysis ===")
        cos_sim = (self.norm_feats * self.tumor_feats).sum(dim=1)
        print(f"1. [Macro] Cosine Similarity (Mean): {cos_sim.mean().item():.4f}")
        print("   -> 解释: 值很小(<0.5)，证明 High/Low 语义完全不同 (这是好事!)")

    def get_eval_conditions(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        c_norm = self.norm_feats.unsqueeze(0).expand(batch_size, -1, -1)
        c_tumor = self.tumor_feats.unsqueeze(0).expand(batch_size, -1, -1)
        return c_norm, c_tumor


class StructuredPatchDataset(Dataset):
    def __init__(self, csv_path: str, feature_dir: str, target_tissues: List[str], topk: int = 200):
        self.df = pd.read_csv(csv_path)
        self.feature_dir = feature_dir
        self.target_tissues = target_tissues
        self.num_tissues = len(target_tissues)
        self.topk = topk
        self.valid_data: List[dict] = []

        for _, row in self.df.iterrows():
            sid = str(row["slide_id"])
            path = os.path.join(feature_dir, f"{sid}.pt")
            if os.path.exists(path):
                self.valid_data.append({"slide_id": sid, "path": path, "label": LABEL_MAP.get(row["label"], 0)})

        print(f"Dataset Loaded: {len(self.valid_data)} samples. Targeting {self.num_tissues} tissues.")

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        item = self.valid_data[idx]
        data = torch.load(item["path"], map_location="cpu")

        raw_features = data["selected_features"]
        raw_types = data["tissue_types"]

        selected_list = []
        for tissue_name in self.target_tissues:
            if tissue_name in raw_types:
                tissue_idx = raw_types.index(tissue_name)
                selected_list.append(raw_features[tissue_idx])
            else:
                selected_list.append(torch.zeros(self.topk, raw_features.shape[-1]))

        feat = torch.stack(selected_list, dim=0)
        mask = (feat.abs().sum(dim=-1) > 0).float()

        return feat.float(), mask, torch.tensor(item["label"]), item["slide_id"]
