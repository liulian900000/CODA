import argparse
import os

import numpy as np
import pandas as pd
import torch
from lifelines.statistics import logrank_test
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from data import SurvivalConditionManager, SurvivalDataset
from models import HazardCalibrator, IntraTissueGatedAttention, LearnableConditionEmbedding, TissueDiffusionModel
from train import diagnose_stage1_samples, diagnose_stage2_samples, plot_km_curve, train_stage1_discrete, eval_stage1_cindex, eval_stage2_survival
from utils import c_index, get_alphas_cumprod


def main():
    parser = argparse.ArgumentParser(description='Evaluate pretrained CODA survival classifier checkpoints')
    parser.add_argument('--text_bin0', type=str, required=True, help='High Risk Text')
    parser.add_argument('--text_bin1', type=str, required=True, help='Low Risk Text')
    parser.add_argument('--feature_dir', type=str, required=True)
    parser.add_argument('--label_csv', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--topk', type=int, default=200)
    parser.add_argument('--num_bins', type=int, default=4)
    parser.add_argument('--diffusion_num_layers', type=int, default=1)
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--fold', type=int, default=0, help='Which fold split to evaluate')
    parser.add_argument('--stage1_ckpt', type=str, required=True)
    parser.add_argument('--stage2_ckpt', type=str, required=True)
    parser.add_argument('--split_csv', type=str, default='', help='Optional explicit val split csv')
    parser.add_argument('--save_dir', type=str, default='./eval_outputs')
    parser.add_argument('--num_samples_diag', type=int, default=25)
    parser.add_argument('--steps', type=int, default=100)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    alphas = get_alphas_cumprod(1000, device)

    cond_mgr = SurvivalConditionManager(args.text_bin0, args.text_bin1, device, num_bins=args.num_bins)
    target_tissues = cond_mgr.target_tissues
    text_dim = cond_mgr.text_dim
    num_tissues = len(target_tissues)

    d0_data = torch.load(args.text_bin0, map_location=device)
    d1_data = torch.load(args.text_bin1, map_location=device)
    v_high = torch.nn.functional.normalize(d0_data['text_features'], p=2, dim=1)
    v_low = torch.nn.functional.normalize(d1_data['text_features'], p=2, dim=1)

    if args.split_csv:
        val_df = pd.read_csv(args.split_csv)
    else:
        full_df = pd.read_csv(args.label_csv)
        uncensored_times = full_df[full_df['censorship'] == 1]['survival_months'].values
        q_list = np.arange(1, args.num_bins) / args.num_bins
        split_points = np.quantile(uncensored_times, q_list)

        def get_bin_idx(t):
            idx = int(np.searchsorted(split_points, t, side='right'))
            return min(max(idx, 0), args.num_bins - 1)

        full_df['bin_label'] = full_df['survival_months'].apply(get_bin_idx)
        valid_indices = [idx for idx, row in full_df.iterrows() if os.path.exists(os.path.join(args.feature_dir, f"{row['slide_id']}.pt"))]
        full_df = full_df.iloc[valid_indices].reset_index(drop=True)
        full_df['patient_id'] = full_df['slide_id'].apply(lambda x: '-'.join(x.split('-')[:3]))
        full_df['stratify_label'] = full_df['censorship'].astype(str) + '_' + full_df['bin_label'].astype(str)
        groups = full_df['patient_id'].values
        y_stratify = full_df['stratify_label'].values
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1)
        splits = list(sgkf.split(full_df, y_stratify, groups=groups))
        _, val_idx = splits[args.fold]
        val_df = full_df.iloc[val_idx].reset_index(drop=True)

    val_ds = SurvivalDataset(val_df, args.feature_dir, target_tissues=target_tissues, topk=args.topk)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    aggregator = IntraTissueGatedAttention(topk=args.topk, num_tissues=num_tissues, num_bins=args.num_bins).to(device)
    diffusion = TissueDiffusionModel(text_dim=text_dim, num_layers=args.diffusion_num_layers).to(device)
    calibrator = HazardCalibrator(num_bins=args.num_bins).to(device)
    cond_embed = LearnableConditionEmbedding(args.num_bins, num_tissues, text_dim, v_high, v_low).to(device)

    aggregator.load_state_dict(torch.load(args.stage1_ckpt, map_location=device))
    ckpt = torch.load(args.stage2_ckpt, map_location=device)
    diffusion.load_state_dict(ckpt['diffusion'])
    calibrator.load_state_dict(ckpt['calibrator'])

    print('\n>>> STAGE 1 EVALUATION')
    s1_cindex = eval_stage1_cindex(aggregator, val_loader, device)
    print(f'Stage 1 C-Index: {s1_cindex:.4f}')
    diagnose_stage1_samples(aggregator, val_loader, device, num_samples=args.num_samples_diag, save_path=os.path.join(args.save_dir, 'stage1_diagnosis.txt'))

    all_z = []
    with torch.no_grad():
        for batch in val_loader:
            x, mask = batch[0], batch[1]
            z, _ = aggregator(x.to(device), mask.to(device))
            all_z.append(z.cpu())
    all_z = torch.cat(all_z, 0)
    global_mean = all_z.mean(0, keepdim=True).to(device)
    global_std = all_z.std(0, keepdim=True).to(device) + 1e-6

    print('\n>>> STAGE 2 EVALUATION')
    s2_cindex, sids, scores, times, events = eval_stage2_survival(
        aggregator,
        diffusion,
        calibrator,
        cond_embed,
        val_loader,
        alphas,
        device,
        global_mean,
        global_std,
        steps=args.steps,
    )
    print(f'Stage 2 C-Index: {s2_cindex:.4f}')
    diagnose_stage2_samples(
        aggregator,
        diffusion,
        val_loader,
        cond_embed,
        alphas,
        device,
        global_mean,
        global_std,
        num_samples=args.num_samples_diag,
        steps=min(args.steps, 100),
        save_path=os.path.join(args.save_dir, 'stage2_diagnosis.txt'),
    )

    median_score = np.median(scores)
    p_value = logrank_test(
        np.asarray(times)[np.asarray(scores) >= median_score],
        np.asarray(times)[np.asarray(scores) < median_score],
        event_observed_A=np.asarray(events)[np.asarray(scores) >= median_score],
        event_observed_B=np.asarray(events)[np.asarray(scores) < median_score],
    ).p_value
    print(f'Log-rank p-value: {p_value:.4e}')

    plot_km_curve(scores, times, events, save_path=os.path.join(args.save_dir, 'km_curve_eval.png'), title='Evaluation Kaplan-Meier Curve')
    pd.DataFrame({'slide_id': sids, 'risk_score': scores, 'time': times, 'event': events}).to_csv(os.path.join(args.save_dir, 'eval_predictions.csv'), index=False)

    overall_c = c_index(torch.tensor(scores), torch.tensor(times), torch.tensor(events))
    print(f'Pooled C-Index Check: {overall_c:.4f}')


if __name__ == '__main__':
    main()
