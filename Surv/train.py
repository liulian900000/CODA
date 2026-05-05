import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import SurvivalConditionManager, SurvivalDataset
from models import HazardCalibrator, IntraTissueGatedAttention, LearnableConditionEmbedding, TissueDiffusionModel
from utils import (
    c_index,
    discrete_time_hazard_nll,
    get_alphas_cumprod,
    hazards_to_risk,
    q_sample,
)


def plot_km_curve(risk_scores, times, events, save_path="km_curve.png", title="Kaplan-Meier Survival Curve"):
    risk_scores = np.asarray(risk_scores)
    times = np.asarray(times)
    events = np.asarray(events)

    median_risk = np.median(risk_scores)
    high_risk_mask = risk_scores >= median_risk
    low_risk_mask = risk_scores < median_risk

    results = logrank_test(
        times[high_risk_mask],
        times[low_risk_mask],
        event_observed_A=events[high_risk_mask],
        event_observed_B=events[low_risk_mask],
    )
    p_value = results.p_value

    plt.figure(figsize=(6, 5), dpi=300)
    ax = plt.subplot(111)
    kmf = KaplanMeierFitter()

    kmf.fit(times[low_risk_mask], events[low_risk_mask], label="Low Risk")
    kmf.plot_survival_function(ax=ax, color="blue", ci_show=False, linewidth=2)

    kmf.fit(times[high_risk_mask], events[high_risk_mask], label="High Risk")
    kmf.plot_survival_function(ax=ax, color="red", ci_show=False, linewidth=2)

    plt.title(title, fontsize=14)
    plt.xlabel("Time (Months)", fontsize=12)
    plt.ylabel("Proportion Surviving", fontsize=12)
    plt.text(
        0.05,
        0.1,
        f"p-value = {p_value:.4e}",
        transform=ax.transAxes,
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )
    plt.legend(loc="upper right", frameon=False)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.ylim([0.0, 1.05])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def diagnose_stage1_samples(aggregator, loader, device, num_samples=5, save_path=None):
    aggregator.eval()
    records = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 6:
                x, mask, time, event, bin_idx, sid = batch
            else:
                x, mask, time, event, bin_idx = batch
            x, mask = x.to(device), mask.to(device)
            _, hazards = aggregator(x, mask)
            risk = hazards_to_risk(hazards, method="cumhaz_last")
            for i in range(x.shape[0]):
                records.append(
                    {
                        "time": time[i].item(),
                        "event": int(event[i].item()),
                        "true_bin": int(bin_idx[i].item()),
                        "hazards": hazards[i].cpu().numpy(),
                        "risk_score": risk[i].item(),
                    }
                )
            if len(records) >= num_samples:
                break

    records = records[:num_samples]
    if not records:
        return records

    k_bins = len(records[0]["hazards"])
    lines = ["\n" + "=" * 70, "  [Stage 1] Sample-Level Diagnostics", "=" * 70]
    header = "  {:>5s} | {:>6s} | {:>4s} | {:>5s} | " + " | ".join([f"h(bin{k})" for k in range(k_bins)]) + " | {:>10s}"
    lines.append(header.format("Idx", "Time", "Evt", "TBin", "RiskScore"))
    lines.append("-" * 70)
    for idx, r in enumerate(records):
        h_str = " | ".join([f"{r['hazards'][k]:.4f}" for k in range(k_bins)])
        lines.append(
            f"  {idx:>5d} | {r['time']:>6.1f} | {r['event']:>4d} | {r['true_bin']:>5d} | {h_str} | {r['risk_score']:>10.4f}"
        )
    lines.append("=" * 70 + "\n")
    text = "\n".join(lines)
    print(text)
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text)
    return records


def diagnose_stage2_samples(aggregator, diffusion, loader, cond_mgr, alphas, device, mean, std, num_samples=5, steps=10, save_path=None):
    aggregator.eval()
    diffusion.eval()
    k_bins = cond_mgr.K
    t_eval = torch.linspace(50, 950, steps).long().to(device)
    records = []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 6:
                x, mask, time, event, bin_idx, sid = batch
            else:
                x, mask, time, event, bin_idx = batch
            x, mask = x.to(device), mask.to(device)
            bsz = x.shape[0]

            z_0, hazards_s1 = aggregator(x, mask)
            z_0 = (z_0 - mean) / std

            all_conds = cond_mgr.get_all_conditions(bsz)
            cond_flat = all_conds.reshape(bsz * k_bins, -1, all_conds.shape[-1])
            mse_accum = torch.zeros(bsz, k_bins, device=device)

            for t_idx in t_eval:
                t_batch = torch.full((bsz,), t_idx, device=device).long()
                noise = torch.randn_like(z_0)
                z_t = q_sample(z_0, t_batch, noise, alphas)
                z_t_rep = z_t.repeat_interleave(k_bins, dim=0)
                t_rep = t_batch.repeat_interleave(k_bins, dim=0)
                noise_rep = noise.repeat_interleave(k_bins, dim=0)
                pred = diffusion.forward_denoise(z_t_rep, t_rep, cond_flat)
                mse_accum += ((pred - noise_rep) ** 2).mean(dim=[1, 2]).view(bsz, k_bins)

            mse_avg = mse_accum / steps
            tau = 0.05
            scores = -mse_avg / tau
            probs = F.softmax(scores, dim=1)
            bin_ids = torch.arange(k_bins, device=device).float()
            risk_s2 = -(probs * bin_ids.unsqueeze(0)).sum(dim=1)
            risk_s1 = hazards_to_risk(hazards_s1, method="cumhaz_last")

            for i in range(bsz):
                records.append(
                    {
                        "time": time[i].item(),
                        "event": int(event[i].item()),
                        "true_bin": int(bin_idx[i].item()),
                        "mse_per_bin": mse_avg[i].cpu().numpy(),
                        "prob_per_bin": probs[i].cpu().numpy(),
                        "risk_s1": risk_s1[i].item(),
                        "risk_s2": risk_s2[i].item(),
                    }
                )
            if len(records) >= num_samples:
                break

    records = records[:num_samples]
    if not records:
        return records

    k_bins = len(records[0]["mse_per_bin"])
    lines = ["\n" + "=" * 120, "  [Stage 2] Sample-Level Diagnostics", "=" * 120]
    for idx, r in enumerate(records):
        lines.append(
            f"  --- Sample {idx} | Time={r['time']:.1f} | Event={r['event']} | TrueBin={r['true_bin']} | RiskS1={r['risk_s1']:.4f} | RiskS2={r['risk_s2']:.4f} ---"
        )
        lines.append("    MSE   per bin: " + "  ".join([f"bin{k}={r['mse_per_bin'][k]:.6f}" for k in range(k_bins)]))
        diffs = [r['mse_per_bin'][k + 1] - r['mse_per_bin'][k] for k in range(k_bins - 1)]
        lines.append("    ΔMSE (k+1 - k): " + "  ".join([f"Δ{k}->{k + 1}={d:+.6f}" for k, d in enumerate(diffs)]))
        lines.append("    Prob  per bin: " + "  ".join([f"bin{k}={r['prob_per_bin'][k]:.4f}" for k in range(k_bins)]))
        best_bin = int(r["mse_per_bin"].argmin())
        lines.append(f"    Best-match bin (min MSE / max Prob): bin{best_bin}  |  True bin: {r['true_bin']}")
        lines.append("")

    lines.append("=" * 120 + "\n")
    text = "\n".join(lines)
    print(text)
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text)
    return records


def train_stage1_discrete(aggregator, loader, opt, device):
    aggregator.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="S1 Discrete", leave=False):
        if len(batch) == 6:
            x, mask, time, event, bin_idx, sid = batch
        else:
            x, mask, time, event, bin_idx = batch
        x, mask = x.to(device), mask.to(device)
        time, event, bin_idx = time.to(device), event.to(device), bin_idx.to(device)

        opt.zero_grad()
        _, hazards = aggregator(x, mask)
        loss = discrete_time_hazard_nll(hazards, bin_idx, event, beta_uncensored=0.3)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def eval_stage1_cindex(aggregator, loader, device):
    aggregator.eval()
    all_scores, all_times, all_events = [], [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 6:
                x, mask, time, event, bin_idx, sid = batch
            else:
                x, mask, time, event, bin_idx = batch
            x, mask = x.to(device), mask.to(device)
            _, hazards = aggregator(x, mask)
            all_scores.append(hazards_to_risk(hazards, method="cumhaz_last").detach().cpu())
            all_times.append(time)
            all_events.append(event)
    return c_index(torch.cat(all_scores), torch.cat(all_times), torch.cat(all_events))


def train_stage2_survival(aggregator, diffusion, calibrator, cond_embed, loader, opt, alphas, device, mean, std, epoch=0, num_t_steps=100):
    aggregator.eval()
    diffusion.train()
    calibrator.train()
    total_loss = 0.0
    t_fixed_points = torch.linspace(50, 950, num_t_steps).long().tolist()
    loop = tqdm(loader, desc=f"S2 Train Ep{epoch + 1}", leave=False)

    for it, batch in enumerate(loop):
        if len(batch) == 6:
            x, patch_mask, time, event, bin_idx, sid = batch
        else:
            x, patch_mask, time, event, bin_idx = batch
        x, patch_mask = x.to(device), patch_mask.to(device)
        time, event, bin_idx = time.to(device), event.to(device), bin_idx.to(device)
        bsz = x.shape[0]

        opt.zero_grad()
        with torch.no_grad():
            z_0, _ = aggregator(x, patch_mask)
            z_0 = (z_0 - mean) / std

        all_conds = cond_embed.get_all_conditions(bsz)
        k_bins = all_conds.shape[1]
        cond_flat = all_conds.reshape(bsz * k_bins, -1, all_conds.shape[-1])
        mse_accum = torch.zeros(bsz, k_bins, device=device)

        for t_val in t_fixed_points:
            t_steps = torch.full((bsz,), t_val, device=device).long()
            noise = torch.randn_like(z_0)
            z_t = q_sample(z_0, t_steps, noise, alphas)
            z_t_rep = z_t.repeat_interleave(k_bins, dim=0)
            t_rep = t_steps.repeat_interleave(k_bins, dim=0)
            noise_rep = noise.repeat_interleave(k_bins, dim=0)
            pred_noise = diffusion.forward_denoise(z_t_rep, t_rep, cond_flat)
            mse_accum += ((pred_noise - noise_rep) ** 2).mean(dim=[1, 2]).view(bsz, k_bins)

        mse_avg = mse_accum / num_t_steps
        scores = -mse_avg
        hazards = calibrator(scores)
        l_discrete = discrete_time_hazard_nll(hazards, bin_idx, event, beta_uncensored=0.3)
        l_diff = torch.min(mse_avg, dim=1).values.mean()

        l_cens = torch.tensor(0.0, device=device)
        mask_censored = event == 0.0
        if mask_censored.any():
            probs_cens = F.softmax(scores[mask_censored], dim=1)
            cens_bins = bin_idx[mask_censored].unsqueeze(1)
            col_idx = torch.arange(k_bins, device=device).unsqueeze(0).expand(probs_cens.size(0), k_bins)
            valid_mask = (col_idx >= cens_bins).float()
            surv_prob = (probs_cens * valid_mask).sum(dim=1).clamp(min=1e-7)
            l_cens = -torch.log(surv_prob).mean()

        loss = l_discrete + 1.0 * l_diff + 0.2 * l_cens
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(diffusion.parameters()) + list(calibrator.parameters()), 1.0)
        opt.step()
        total_loss += loss.item()
        loop.set_postfix(NLL=f"{l_discrete.item():.4f}", Diff=f"{l_diff.item():.4f}", Cens=f"{l_cens.item():.4f}")
    return total_loss / max(len(loader), 1)


def eval_stage2_survival(aggregator, diffusion, calibrator, cond_embed, loader, alphas, device, mean, std, steps=10):
    aggregator.eval()
    diffusion.eval()
    calibrator.eval()

    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    all_scores, all_times, all_events, all_sids = [], [], [], []
    t_eval = torch.linspace(50, 950, steps).long().to(device)
    k_bins = cond_embed.K

    with torch.no_grad():
        for batch in tqdm(loader, desc="S2 Eval", leave=False):
            x, mask, time, event, bin_idx, sid = batch
            x, mask = x.to(device), mask.to(device)
            bsz = x.shape[0]
            z_0, _ = aggregator(x, mask)
            z_0 = (z_0 - mean) / std

            all_conds = cond_embed.get_all_conditions(bsz)
            cond_flat = all_conds.reshape(bsz * k_bins, -1, all_conds.shape[-1])
            mse_accum = torch.zeros(bsz, k_bins, device=device)

            for t_idx in t_eval:
                t_batch = torch.full((bsz,), t_idx, device=device).long()
                noise = torch.randn_like(z_0)
                z_t = q_sample(z_0, t_batch, noise, alphas)
                z_t_rep = z_t.repeat_interleave(k_bins, dim=0)
                t_rep = t_batch.repeat_interleave(k_bins, dim=0)
                noise_rep = noise.repeat_interleave(k_bins, dim=0)
                pred = diffusion.forward_denoise(z_t_rep, t_rep, cond_flat)
                mse_accum += ((pred - noise_rep) ** 2).mean(dim=[1, 2]).view(bsz, k_bins)

            mse_avg = mse_accum / steps
            hazards = calibrator(-mse_avg)
            risk_scores = hazards_to_risk(hazards, method="cumhaz_last")

            all_scores.extend(risk_scores.cpu().numpy().tolist())
            all_times.extend(time.numpy().tolist())
            all_events.extend(event.numpy().tolist())
            all_sids.extend(list(sid))

    c_idx = c_index(torch.tensor(all_scores), torch.tensor(all_times), torch.tensor(all_events))
    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state_all(cuda_rng_state)
    return c_idx, all_sids, all_scores, all_times, all_events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_bin0', type=str, required=True, help='High Risk Text')
    parser.add_argument('--text_bin1', type=str, required=True, help='Low Risk Text')
    parser.add_argument('--feature_dir', type=str, required=True)
    parser.add_argument('--label_csv', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs_stage1', type=int, default=30)
    parser.add_argument('--epochs_stage2', type=int, default=50)
    parser.add_argument('--topk', type=int, default=200)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_bins', type=int, default=4)
    parser.add_argument('--diffusion_num_layers', type=int, default=1)
    parser.add_argument('--split_save_dir', type=str, default='./splits')
    parser.add_argument('--ckpt_save_dir', type=str, default='.')
    parser.add_argument('--device', type=str, default='')
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    alphas = get_alphas_cumprod(1000, device)
    os.makedirs(args.split_save_dir, exist_ok=True)
    os.makedirs(args.ckpt_save_dir, exist_ok=True)

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
    full_df = full_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    full_df['stratify_label'] = full_df['censorship'].astype(str) + '_' + full_df['bin_label'].astype(str)

    groups = full_df['patient_id'].values
    y_stratify = full_df['stratify_label'].values
    sgkf = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    fold_results, fold_p_values = [], []
    overall_val_scores, overall_val_times, overall_val_events = [], [], []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(full_df, y_stratify, groups=groups)):
        print(f"\n{'=' * 60}\nFOLD {fold + 1}/{args.n_folds}\n{'=' * 60}")
        train_df = full_df.iloc[train_idx].reset_index(drop=True)
        val_df = full_df.iloc[val_idx].reset_index(drop=True)
        train_df.to_csv(os.path.join(args.split_save_dir, f"train_fold{fold}.csv"), index=False)
        val_df.to_csv(os.path.join(args.split_save_dir, f"val_fold{fold}.csv"), index=False)

        cond_mgr = SurvivalConditionManager(args.text_bin0, args.text_bin1, device, num_bins=args.num_bins)
        target_tissues = cond_mgr.target_tissues
        text_dim = cond_mgr.text_dim
        num_tissues = len(target_tissues)

        d0_data = torch.load(args.text_bin0, map_location=device)
        d1_data = torch.load(args.text_bin1, map_location=device)
        v_high = F.normalize(d0_data['text_features'], p=2, dim=1)
        v_low = F.normalize(d1_data['text_features'], p=2, dim=1)

        train_ds = SurvivalDataset(train_df, args.feature_dir, target_tissues=target_tissues, topk=args.topk)
        val_ds = SurvivalDataset(val_df, args.feature_dir, target_tissues=target_tissues, topk=args.topk)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        aggregator = IntraTissueGatedAttention(topk=args.topk, num_tissues=num_tissues, num_bins=args.num_bins).to(device)
        diffusion = TissueDiffusionModel(text_dim=text_dim, num_layers=args.diffusion_num_layers).to(device)
        calibrator = HazardCalibrator(num_bins=args.num_bins).to(device)
        cond_embed = LearnableConditionEmbedding(args.num_bins, num_tissues, text_dim, v_high, v_low).to(device)

        opt1 = optim.Adam(aggregator.parameters(), lr=2e-4, weight_decay=1e-5)
        best_c1 = 0.0
        best_s1_path = os.path.join(args.ckpt_save_dir, f"best_s1_fold{fold}.pth")
        for ep in range(args.epochs_stage1):
            l = train_stage1_discrete(aggregator, train_loader, opt1, device)
            c = eval_stage1_cindex(aggregator, val_loader, device)
            print(f"S1 Ep {ep + 1}: Loss {l:.4f} | Val C-Index {c:.4f}")
            if c > best_c1:
                best_c1 = c
                torch.save(aggregator.state_dict(), best_s1_path)

        if os.path.exists(best_s1_path):
            aggregator.load_state_dict(torch.load(best_s1_path, map_location=device))
        diagnose_stage1_samples(aggregator, val_loader, device, num_samples=205, save_path=os.path.join(args.split_save_dir, f"diag_stage1_fold{fold}.txt"))

        all_z = []
        aggregator.eval()
        with torch.no_grad():
            for batch in train_loader:
                x, mask = batch[0], batch[1]
                z, _ = aggregator(x.to(device), mask.to(device))
                all_z.append(z.cpu())
        all_z = torch.cat(all_z, 0)
        global_mean = all_z.mean(0, keepdim=True).to(device)
        global_std = all_z.std(0, keepdim=True).to(device) + 1e-6

        for p in aggregator.parameters():
            p.requires_grad = False

        opt2 = optim.Adam(list(diffusion.parameters()) + list(calibrator.parameters()), lr=1e-4, weight_decay=1e-3)
        best_c2 = 0.0
        best_s2_path = os.path.join(args.ckpt_save_dir, f"best_s2_fold{fold}.pth")
        best_fold_scores = best_fold_times = best_fold_events = best_fold_sids = None

        for ep in range(args.epochs_stage2):
            l = train_stage2_survival(aggregator, diffusion, calibrator, cond_embed, train_loader, opt2, alphas, device, global_mean, global_std, epoch=ep)
            c, sids_val, scores_val, times_val, events_val = eval_stage2_survival(aggregator, diffusion, calibrator, cond_embed, val_loader, alphas, device, global_mean, global_std)
            print(f"S2 Ep {ep + 1}: Loss {l:.4f} | Val C-Index {c:.4f}")
            if c > best_c2:
                best_c2 = c
                torch.save({"diffusion": diffusion.state_dict(), "calibrator": calibrator.state_dict()}, best_s2_path)
                best_fold_scores, best_fold_times, best_fold_events, best_fold_sids = scores_val, times_val, events_val, sids_val
                plot_km_curve(scores_val, times_val, events_val, save_path=os.path.join(args.split_save_dir, f"km_curve_fold{fold}.png"), title=f"Fold {fold} Validation KM Curve (C-index: {c:.3f})")
                pd.DataFrame({"fold": fold, "epoch": ep + 1, "slide_id": sids_val, "risk_score": scores_val, "time": times_val, "event": events_val}).to_csv(os.path.join(args.split_save_dir, f"val_predictions_fold{fold}_best.csv"), index=False)

        if os.path.exists(best_s2_path):
            ckpt = torch.load(best_s2_path, map_location=device)
            diffusion.load_state_dict(ckpt["diffusion"])
            calibrator.load_state_dict(ckpt["calibrator"])
        diagnose_stage2_samples(aggregator, diffusion, val_loader, cond_embed, alphas, device, global_mean, global_std, num_samples=205, steps=100, save_path=os.path.join(args.split_save_dir, f"diag_stage2_fold{fold}.txt"))

        if best_fold_scores is None:
            best_fold_scores, best_fold_times, best_fold_events, best_fold_sids = scores_val, times_val, events_val, sids_val

        median_score = np.median(best_fold_scores)
        fold_p = logrank_test(
            np.asarray(best_fold_times)[np.asarray(best_fold_scores) >= median_score],
            np.asarray(best_fold_times)[np.asarray(best_fold_scores) < median_score],
            event_observed_A=np.asarray(best_fold_events)[np.asarray(best_fold_scores) >= median_score],
            event_observed_B=np.asarray(best_fold_events)[np.asarray(best_fold_scores) < median_score],
        ).p_value

        print(f"Fold {fold + 1} Best Val C-Index: {best_c2:.4f} | p-value: {fold_p:.4e}")
        fold_results.append(best_c2)
        fold_p_values.append(fold_p)
        overall_val_scores.extend(best_fold_scores)
        overall_val_times.extend(best_fold_times)
        overall_val_events.extend(best_fold_events)
        pd.DataFrame({"fold": fold, "slide_id": best_fold_sids, "risk_score": best_fold_scores, "time": best_fold_times, "event": best_fold_events}).to_csv(os.path.join(args.split_save_dir, f"val_predictions_fold{fold}.csv"), index=False)

    print("\n" + "=" * 60 + "\nCROSS-VALIDATION RESULTS (Val C-Index)\n" + "=" * 60)
    for i, c in enumerate(fold_results):
        print(f"Fold {i + 1}: {c:.4f}")
    print(f"Mean Val C-Index: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
    print(f"Mean p-value: {np.mean(fold_p_values):.4e} ± {np.std(fold_p_values):.4e}")

    overall_c = c_index(torch.tensor(overall_val_scores), torch.tensor(overall_val_times), torch.tensor(overall_val_events))
    print(f"Overall Pooled C-Index: {overall_c:.4f}")
    plot_km_curve(overall_val_scores, overall_val_times, overall_val_events, save_path=os.path.join(args.split_save_dir, "km_curve_overall.png"), title="Overall Pooled Validation KM Curve")


if __name__ == '__main__':
    main()
