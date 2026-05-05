import os
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class SurvivalSample:
    path: str
    time: float
    event: int
    sid: str
    bin: int


class SurvivalConditionManager:
    def __init__(self, bin0_path, bin1_path, device, num_bins=4):
        self.device = device
        d0_data = torch.load(bin0_path, map_location=device)
        d1_data = torch.load(bin1_path, map_location=device)

        self.target_tissues = d0_data["tissue_types"]
        v_high = torch.nn.functional.normalize(d0_data["text_features"], p=2, dim=1)
        v_low = torch.nn.functional.normalize(d1_data["text_features"], p=2, dim=1)

        diff = (v_high - v_low).abs().mean().item()
        print(f"\n[ConditionManager] Target Tissues: {self.target_tissues}")
        print(f"[DBG text] Abs Diff Mean: {diff:.8f}")

        self.conds = []
        self.num_bins = num_bins
        for i in range(num_bins):
            alpha = i / (num_bins - 1) if num_bins > 1 else 0
            v_mixed = (1 - alpha) * v_high + alpha * v_low
            v_mixed = torch.nn.functional.normalize(v_mixed, p=2, dim=1)
            self.conds.append(v_mixed)

        self.text_dim = self.conds[0].shape[-1]
        self.K = len(self.conds)

    def get_all_conditions(self, batch_size):
        all_c = torch.stack(self.conds, dim=0)
        return all_c.unsqueeze(0).expand(batch_size, -1, -1, -1)


class SurvivalDataset(Dataset):
    def __init__(self, df_split, feature_dir, target_tissues, topk=200):
        self.feature_dir = feature_dir
        self.target_tissues = target_tissues
        self.num_tissues = len(target_tissues)
        self.topk = topk
        self.valid_data = []

        for _, row in df_split.iterrows():
            sid = str(row["slide_id"])
            path = os.path.join(feature_dir, f"{sid}.pt")
            if os.path.exists(path):
                self.valid_data.append({
                    "path": path,
                    "time": float(row["survival_months"]),
                    "event": int(row["censorship"]),
                    "sid": sid,
                    "bin": int(row["bin_label"]),
                })

        print(f"Dataset Loaded: {len(self.valid_data)} samples. Targeting {self.num_tissues} tissues.")

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        item = self.valid_data[idx]
        data = torch.load(item["path"], map_location="cpu")

        if isinstance(data, dict):
            raw_features = data["selected_features"]
            raw_types = data["tissue_types"]
            selected_list = []
            for t_name in self.target_tissues:
                if t_name in raw_types:
                    t_idx = raw_types.index(t_name)
                    selected_list.append(raw_features[t_idx])
                else:
                    selected_list.append(torch.zeros(self.topk, raw_features.shape[-1]))
            feat = torch.stack(selected_list, dim=0)
        else:
            feat = data
            if feat.dim() == 2:
                feat = feat.view(self.num_tissues, -1, feat.shape[-1])

        n, k_curr, d = feat.shape
        if k_curr > self.topk:
            feat = feat[:, : self.topk, :]
        elif k_curr < self.topk:
            pad = self.topk - k_curr
            feat = torch.nn.functional.pad(feat, (0, 0, 0, pad), "constant", 0)

        mask = (feat.abs().sum(dim=-1) > 0).float()
        return (
            feat.float(),
            mask,
            torch.tensor(item["time"], dtype=torch.float32),
            torch.tensor(item["event"], dtype=torch.float32),
            torch.tensor(item["bin"], dtype=torch.long),
            item["sid"],
        )
