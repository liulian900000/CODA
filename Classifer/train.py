import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from data import ConditionManager, StructuredPatchDataset
from models import IntraTissueGatedAttention, TissueDiffusionModel
from utils import get_alphas_cumprod, q_sample


def train_stage1_epoch(aggregator, loader, opt, device):
    aggregator.train()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    for x, mask, y, _ in tqdm(loader, desc='S1 Train', leave=False):
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        opt.zero_grad()
        z_0 = aggregator(x, mask)
        logits = aggregator.get_logits(z_0)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_stage1(aggregator, loader, device, save_csv=None):
    aggregator.eval()
    y_true, y_probs = [], []
    all_sids, all_probs_full = [], []
    with torch.no_grad():
        for x, mask, y, sids in loader:
            x, mask = x.to(device), mask.to(device)
            z_0 = aggregator(x, mask)
            probs = torch.softmax(aggregator.get_logits(z_0), dim=1)
            y_true.extend(y.numpy())
            y_probs.extend(probs[:, 1].cpu().numpy())
            all_sids.extend(sids)
            all_probs_full.extend(probs.cpu().numpy())

    y_true, y_probs = np.array(y_true), np.array(y_probs)
    all_probs_full = np.array(all_probs_full)
    auc = roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.0
    y_pred = (y_probs > 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    if save_csv is not None:
        df = pd.DataFrame({
            'slide_id': all_sids,
            'true_label': ['normal' if t == 0 else 'tumor' for t in y_true],
            'prob_normal': all_probs_full[:, 0],
            'prob_tumor': all_probs_full[:, 1],
            'predicted': ['tumor' if p > 0.5 else 'normal' for p in y_probs],
            'correct': [int(y_pred[i] == int(y_true[i])) for i in range(len(y_true))],
        })
        df.to_csv(save_csv, index=False)

    return {'metrics': {'auc': auc, 'acc': acc, 'f1': f1, 'recall': recall, 'specificity': spec}, 'raw': {'y_true': y_true, 'y_scores': y_probs}}


def train_stage2_epoch(aggregator, diffusion, loader, opt, cond_mgr, alphas, device, mean, std, tau=0.1, lambda_cls=0.5, lambda_rank=0.2, rank_margin=0.02):
    aggregator.eval()
    diffusion.train()
    total_loss = 0
    for x, mask, y, _ in tqdm(loader, desc='S2 Train', leave=False):
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        b = x.shape[0]
        opt.zero_grad()
        with torch.no_grad():
            z_0 = (aggregator(x, mask) - mean) / std
        t = torch.randint(0, 600, (b,), device=device).long()
        noise = torch.randn_like(z_0)
        z_t = q_sample(z_0, t, noise, alphas)
        c_norm, c_tumor = cond_mgr.get_eval_conditions(b)
        pred_n = diffusion.forward_denoise(z_t, t, c_norm)
        pred_t = diffusion.forward_denoise(z_t, t, c_tumor)
        mse_n = ((pred_n - noise) ** 2).mean(dim=[1, 2])
        mse_t = ((pred_t - noise) ** 2).mean(dim=[1, 2])
        loss_mse = torch.where(y == 0, mse_n, mse_t).mean()
        diff_raw = mse_n - mse_t
        loss_cls = F.binary_cross_entropy_with_logits(diff_raw / tau, y.float())
        s = y.float() * 2 - 1
        loss_rank = F.relu(rank_margin - s * diff_raw).mean()
        loss = loss_mse + loss_cls * lambda_cls + loss_rank * lambda_rank
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_stage2_likelihood(aggregator, diffusion, loader, cond_mgr, alphas, device, mean, std, steps=50, fixed_thresh=None, save_csv=None, threshold_mode='best_f1'):
    rng_state = torch.get_rng_state()
    torch.manual_seed(42)
    aggregator.eval()
    diffusion.eval()
    y_true, y_scores = [], []
    all_sids, all_err_norm, all_err_tumor = [], [], []
    t_eval = torch.linspace(0, 200, steps).round().long().unique().to(device)
    num_eval_steps = len(t_eval)

    with torch.no_grad():
        for x, mask, y, sids in tqdm(loader, desc='S2 Eval', leave=False):
            x, mask = x.to(device), mask.to(device)
            b = x.shape[0]
            z_0 = (aggregator(x, mask) - mean) / std
            c_norm, c_tumor = cond_mgr.get_eval_conditions(b)
            err_norm, err_tumor = 0, 0
            gen = torch.Generator(device=device)
            gen.manual_seed(42)
            for t_idx in t_eval:
                t_batch = torch.full((b,), t_idx, device=device).long()
                noise = torch.randn(z_0.shape, generator=gen, device=device)
                z_t = q_sample(z_0, t_batch, noise, alphas)
                p_n = diffusion.forward_denoise(z_t, t_batch, c_norm)
                p_t = diffusion.forward_denoise(z_t, t_batch, c_tumor)
                err_norm += ((p_n - noise) ** 2).mean(dim=[1, 2])
                err_tumor += ((p_t - noise) ** 2).mean(dim=[1, 2])
            err_norm = err_norm / num_eval_steps
            err_tumor = err_tumor / num_eval_steps
            y_scores.extend((err_norm - err_tumor).cpu().numpy())
            y_true.extend(y.numpy())
            all_sids.extend(sids)
            all_err_norm.extend(err_norm.cpu().numpy())
            all_err_tumor.extend(err_tumor.cpu().numpy())

    torch.set_rng_state(rng_state)
    y_true, y_scores = np.array(y_true), np.array(y_scores)
    auc = roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.0

    if fixed_thresh is not None:
        yp = (y_scores > fixed_thresh).astype(int)
        best_thresh = fixed_thresh
        best_f1 = f1_score(y_true, yp, zero_division=0)
        best_acc = accuracy_score(y_true, yp)
        best_recall = recall_score(y_true, yp, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, yp).ravel()
        best_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        best_f1, best_acc, best_thresh = -1.0, -1.0, 0
        best_recall, best_spec = 0, 0
        thresholds = np.percentile(y_scores, np.linspace(0, 100, 100))
        for th in thresholds:
            yp = (y_scores > th).astype(int)
            f = f1_score(y_true, yp, zero_division=0)
            acc = accuracy_score(y_true, yp)
            better = acc > best_acc if threshold_mode == 'best_acc' else f > best_f1
            if better:
                best_f1, best_acc, best_thresh = f, acc, th
                best_recall = recall_score(y_true, yp, zero_division=0)
                tn, fp, fn, tp = confusion_matrix(y_true, yp).ravel()
                best_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    if save_csv is not None:
        yp_final = (y_scores > best_thresh).astype(int)
        df = pd.DataFrame({
            'slide_id': all_sids,
            'true_label': ['normal' if t == 0 else 'tumor' for t in y_true],
            'err_norm': all_err_norm,
            'err_tumor': all_err_tumor,
            'diff_score': y_scores,
            'threshold': best_thresh,
            'predicted': ['tumor' if p == 1 else 'normal' for p in yp_final],
            'correct': [int(yp_final[i] == int(y_true[i])) for i in range(len(y_true))],
        })
        df.to_csv(save_csv, index=False)

    return {'metrics': {'auc': auc, 'acc': best_acc, 'f1': best_f1, 'recall': best_recall, 'specificity': best_spec, 'best_thresh': best_thresh}, 'raw': {'y_true': y_true, 'y_scores': y_scores}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_dir', type=str, required=True)
    parser.add_argument('--label_csv', type=str, required=True)
    parser.add_argument('--normal_text_pt', type=str, required=True)
    parser.add_argument('--abnormal_text_pt', type=str, required=True)
    parser.add_argument('--topk', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs_stage1', type=int, default=100)
    parser.add_argument('--epochs_stage2', type=int, default=150)
    parser.add_argument('--eval_steps', type=int, default=100)
    parser.add_argument('--lambda_cls', type=float, default=0.7)
    parser.add_argument('--lambda_rank', type=float, default=0.2)
    parser.add_argument('--rank_margin', type=float, default=0.02)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    cond_mgr = ConditionManager(args.normal_text_pt, args.abnormal_text_pt, device)
    alphas = get_alphas_cumprod(1000, device)
    target_tissues = cond_mgr.target_tissues

    full_ds = StructuredPatchDataset(args.label_csv, args.feature_dir, target_tissues=target_tissues, topk=args.topk)
    test_idxs = [i for i, d in enumerate(full_ds.valid_data) if d['slide_id'].startswith('test')]
    train_val_idxs = [i for i, d in enumerate(full_ds.valid_data) if not d['slide_id'].startswith('test')]
    tv_labels = [full_ds.valid_data[i]['label'] for i in train_val_idxs]
    train_idxs, val_idxs = train_test_split(train_val_idxs, test_size=0.2, stratify=tv_labels, random_state=42)

    def get_loader(idxs, shuffle=False, weighted=False):
        if len(idxs) == 0:
            return None
        ds = torch.utils.data.Subset(full_ds, idxs)
        if weighted:
            lbls = [full_ds.valid_data[i]['label'] for i in idxs]
            w = 1. / np.bincount(lbls)
            sw = [w[l] for l in lbls]
            sampler = WeightedRandomSampler(sw, len(sw))
            return DataLoader(ds, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)

    train_loader = get_loader(train_idxs, weighted=True)
    val_loader = get_loader(val_idxs)
    test_loader = get_loader(test_idxs)

    aggregator = IntraTissueGatedAttention(num_tissues=len(target_tissues), topk=args.topk).to(device)
    diffusion = TissueDiffusionModel(num_layers=2).to(device)

    print('\n>>> STAGE 1: Training Aggregator')
    opt_agg = optim.Adam(aggregator.parameters(), lr=1e-4, weight_decay=1e-4)
    best_mil_auc = 0.0
    for ep in range(args.epochs_stage1):
        loss = train_stage1_epoch(aggregator, train_loader, opt_agg, device)
        if (ep + 1) % 5 == 0:
            val_auc = eval_stage1(aggregator, val_loader, device)['metrics']['auc']
            print(f'S1 Ep {ep + 1} | Loss: {loss:.4f} | Val AUC: {val_auc:.4f}')
            if val_auc > best_mil_auc:
                best_mil_auc = val_auc
                torch.save(aggregator.state_dict(), 'best_stage1_agg_more.pth')

    aggregator.load_state_dict(torch.load('best_stage1_agg_more.pth'))
    aggregator.eval()

    all_z = []
    with torch.no_grad():
        for x, mask, _, _ in train_loader:
            all_z.append(aggregator(x.to(device), mask.to(device)).cpu())
    all_z = torch.cat(all_z, dim=0)
    global_mean = all_z.mean(dim=0, keepdim=True).to(device)
    global_std = all_z.std(dim=0, keepdim=True).to(device) + 1e-6

    print('\n>>> STAGE 2: Training Diffusion (Contrastive)')
    for param in aggregator.parameters():
        param.requires_grad = False

    opt_diff = optim.Adam(diffusion.parameters(), lr=1e-4, weight_decay=1e-4)
    best_diff_auc = 0.0
    best_val_thresh = 0.0
    best_val_metrics = None
    for ep in range(args.epochs_stage2):
        loss = train_stage2_epoch(aggregator, diffusion, train_loader, opt_diff, cond_mgr, alphas, device, global_mean, global_std, lambda_cls=args.lambda_cls, lambda_rank=args.lambda_rank, rank_margin=args.rank_margin)
        metrics = eval_stage2_likelihood(aggregator, diffusion, val_loader, cond_mgr, alphas, device, global_mean, global_std, steps=args.eval_steps)
        val_metrics = metrics['metrics']
        print(f'S2 Ep {ep + 1} | Loss: {loss:.4f} | Val AUC: {val_metrics["auc"]:.4f}')
        if val_metrics['auc'] > best_diff_auc:
            best_diff_auc = val_metrics['auc']
            best_val_thresh = val_metrics['best_thresh']
            best_val_metrics = metrics
            torch.save(diffusion.state_dict(), 'best_stage2_diff_more.pth')

    if test_loader:
        print('\n' + '=' * 40)
        print('>>> FINAL EVALUATION')
        print('=' * 40)
        if best_val_metrics is not None:
            ref = best_val_metrics['metrics']
            print('>>> [Reference] Best Validation Metrics (Model Selection Basis):')
            print(f'    Val AUC:         {ref["auc"]:.4f}')
            print(f'    Val Acc:         {ref["acc"]:.4f}')
            print(f'    Val F1:          {ref["f1"]:.4f}')
            print(f'    Val Recall:      {ref["recall"]:.4f}')
            print(f'    Val Specificity: {ref["specificity"]:.4f}')
            print('-' * 40)

        eval_stage1(aggregator, test_loader, device, save_csv='stage1_test_diagnosis.csv')
        diffusion.load_state_dict(torch.load('best_stage2_diff_more.pth'))
        m = eval_stage2_likelihood(aggregator, diffusion, test_loader, cond_mgr, alphas, device, global_mean, global_std, steps=100, fixed_thresh=best_val_thresh, save_csv='stage2_test_diagnosis.csv')['metrics']
        print('>>> [Result] Final Test Set Performance:')
        print(f'    Test AUC:         {m["auc"]:.4f}')
        print(f'    Test Acc:         {m["acc"]:.4f}')
        print(f'    Test F1:          {m["f1"]:.4f}')
        print(f'    Test Recall:      {m["recall"]:.4f}')
        print(f'    Test Specificity: {m["specificity"]:.4f}')
        print('=' * 40 + '\n')


if __name__ == '__main__':
    main()
