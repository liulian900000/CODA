import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt


def get_alphas_cumprod(num_timesteps: int = 1000, device: str | torch.device = "cuda") -> torch.Tensor:
    scale = 1000 / num_timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)


def q_sample(z_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, alphas_cumprod: torch.Tensor) -> torch.Tensor:
    alpha_bar = alphas_cumprod[t].view(-1, 1, 1)
    return torch.sqrt(alpha_bar) * z_0 + torch.sqrt(1.0 - alpha_bar) * noise


def cox_ph_loss(risk: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    risk = risk.reshape(-1)
    times = times.reshape(-1)
    events = events.reshape(-1).float()

    num_events = events.sum()
    if num_events.item() == 0:
        return risk.sum() * 0

    order = torch.argsort(times, descending=True)
    risk_ord = risk[order]
    events_ord = events[order]
    log_cumsum_exp = torch.logcumsumexp(risk_ord, dim=0)
    diff = risk_ord - log_cumsum_exp
    loss = -(diff * events_ord).sum() / num_events.clamp(min=1.0)
    return loss


def c_index(scores: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> float:
    scores = scores.detach().reshape(-1).cpu()
    times = times.detach().reshape(-1).cpu()
    events = events.detach().reshape(-1).cpu()

    n = scores.numel()
    concordant = 0.0
    comparable = 0.0
    for i in range(n):
        if int(events[i].item()) != 1:
            continue
        mask = times > times[i]
        if not bool(mask.any()):
            continue
        s_i = scores[i]
        s_j = scores[mask]
        comparable += float(s_j.numel())
        concordant += float((s_i > s_j).sum().item())
        concordant += 0.5 * float((s_i == s_j).sum().item())
    return float(concordant / comparable) if comparable > 0 else 0.0


def hazards_to_log_survival(hazard: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    h = hazard.clamp(min=eps, max=1.0 - eps)
    return torch.cumsum(torch.log1p(-h), dim=1)


def hazards_to_risk(hazard: torch.Tensor, eps: float = 1e-7, method: str = "cumhaz_last") -> torch.Tensor:
    logS = hazards_to_log_survival(hazard, eps=eps)
    if method in {"cumhaz_last", "neg_logS_last"}:
        return -logS[:, -1]
    if method == "sum_hazard":
        return hazard.sum(dim=1)
    raise ValueError(f"Unknown risk method: {method}")


def discrete_time_hazard_nll(hazards, labels, events, weights=None, beta_uncensored=0.0):
    eps = 1e-7
    hazards = torch.clamp(hazards, min=eps, max=1.0 - eps)
    survival = torch.cumprod(1.0 - hazards, dim=1)
    S_padded = torch.cat([torch.ones_like(survival[:, :1]), survival], dim=1)
    batch_range = torch.arange(hazards.size(0), device=hazards.device)

    S_prev_y = S_padded[batch_range, labels]
    S_y = S_padded[batch_range, labels + 1]
    h_y = hazards[batch_range, labels]

    log_likelihood = events * (torch.log(S_prev_y + eps) + torch.log(h_y + eps)) + (1.0 - events) * torch.log(S_y + eps)
    loss = -log_likelihood

    if beta_uncensored > 0:
        loss = loss * (1.0 + beta_uncensored * events)
    if weights is not None:
        loss = loss * weights
    return loss.mean()


def asymmetric_contrastive_recon_loss(mse_avg, bins, event, base_margin=0.05, alpha=2.0):
    mask = (event == 1.0)
    if not mask.any():
        return mse_avg.sum() * 0.0

    mse_u = mse_avg[mask]
    bins_u = bins[mask]
    K = mse_u.shape[1]

    mse_true = mse_u.gather(1, bins_u.view(-1, 1))
    target_bins = torch.arange(K, device=mse_avg.device).view(1, K)
    margin_matrix = torch.where(target_bins > bins_u.view(-1, 1), base_margin * alpha, base_margin)
    diff = F.relu(mse_true - mse_u + margin_matrix)

    false_mask = torch.ones_like(mse_u, dtype=torch.bool)
    false_mask.scatter_(1, bins_u.view(-1, 1), False)
    return (diff * false_mask).sum() / max(false_mask.sum().item(), 1)


def inspect_text_conditions(bin0_path, bin1_path, device, num_bins=4, save_path=None):
    d0 = torch.load(bin0_path, map_location=device)
    d1 = torch.load(bin1_path, map_location=device)

    tissue_types = d0["tissue_types"]
    v_high = F.normalize(d0["text_features"].float(), p=2, dim=1)
    v_low = F.normalize(d1["text_features"].float(), p=2, dim=1)

    assert v_high.shape == v_low.shape, "text feature shape mismatch"
    assert tissue_types == d1["tissue_types"], "tissue type order mismatch"

    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("[Inspect Text Conditions]")
    lines.append("=" * 100)

    cos_per_tissue = F.cosine_similarity(v_high, v_low, dim=1)
    l2_per_tissue = torch.norm(v_high - v_low, dim=1)

    lines.append("\n[Endpoint: bin0 vs bin1]")
    lines.append(f"Global mean cosine similarity: {cos_per_tissue.mean().item():.6f}")
    lines.append(f"Global mean L2 distance     : {l2_per_tissue.mean().item():.6f}")
    lines.append(f"Abs diff mean              : {(v_high - v_low).abs().mean().item():.6f}")

    lines.append("\nPer-tissue endpoint difference:")
    for i, t in enumerate(tissue_types):
        lines.append(f"  {i:02d} | {t:20s} | cosine={cos_per_tissue[i].item():.6f} | l2={l2_per_tissue[i].item():.6f}")

    conds = []
    alphas = []
    for i in range(num_bins):
        alpha = i / (num_bins - 1) if num_bins > 1 else 0.0
        alphas.append(alpha)
        v = (1 - alpha) * v_high + alpha * v_low
        v = F.normalize(v, p=2, dim=1)
        conds.append(v)

    cond_global = torch.stack([c.reshape(-1) for c in conds], dim=0)
    cond_global = F.normalize(cond_global, p=2, dim=1)
    sim_matrix = cond_global @ cond_global.t()

    lines.append("\n[Interpolated condition similarity matrix]")
    header = "        " + "  ".join([f"bin{k}" for k in range(num_bins)])
    lines.append(header)
    for i in range(num_bins):
        row = "  " + f"bin{i}".ljust(5) + " " + "  ".join([f"{sim_matrix[i, j].item():.4f}" for j in range(num_bins)])
        lines.append(row)

    lines.append("\n[Per-tissue cosine to endpoints across interpolated bins]")
    for ti, t in enumerate(tissue_types):
        lines.append(f"\nTissue: {t}")
        lines.append("  bin_k   alpha     cos(to bin0 endpoint)   cos(to bin1 endpoint)")
        for k, c in enumerate(conds):
            c_t = c[ti].unsqueeze(0)
            ch = v_high[ti].unsqueeze(0)
            cl = v_low[ti].unsqueeze(0)
            cos_to_high = F.cosine_similarity(c_t, ch).item()
            cos_to_low = F.cosine_similarity(c_t, cl).item()
            lines.append(f"  bin{k:<5d} {alphas[k]:.3f}         {cos_to_high:.6f}               {cos_to_low:.6f}")

    text = "\n".join(lines)
    print(text)

    if save_path is not None:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[Inspect Text] saved to {save_path}")

    return {"tissue_types": tissue_types, "endpoint_cosine": cos_per_tissue.cpu(), "endpoint_l2": l2_per_tissue.cpu(), "sim_matrix": sim_matrix.cpu()}


def inspect_text_vs_stage1_features(aggregator, loader, cond_embed, device, mean, std, num_batches=3, save_path=None):
    aggregator.eval()
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("[Inspect Text Conditions vs Stage1 z0]")
    lines.append("=" * 100)

    all_sample_stats = []
    with torch.no_grad():
        for bidx, (x, mask, time, event, bin_idx) in enumerate(loader):
            if bidx >= num_batches:
                break
            x, mask = x.to(device), mask.to(device)
            z0, _ = aggregator(x, mask)
            z0 = (z0 - mean) / std
            z0 = F.normalize(z0, p=2, dim=-1)
            B = z0.shape[0]
            all_conds = cond_embed.get_all_conditions(B)
            all_conds = F.normalize(all_conds, p=2, dim=-1)
            K = all_conds.shape[1]
            for i in range(B):
                sims = []
                for k in range(K):
                    sim_k = F.cosine_similarity(z0[i], all_conds[i, k], dim=-1).mean().item()
                    sims.append(sim_k)
                best_bin = int(np.argmax(sims))
                all_sample_stats.append({"time": float(time[i].item()), "event": int(event[i].item()), "true_bin": int(bin_idx[i].item()), "sim_per_bin": sims, "best_bin": best_bin})

    for idx, r in enumerate(all_sample_stats[:100]):
        sim_str = "  ".join([f"bin{k}={r['sim_per_bin'][k]:.4f}" for k in range(len(r['sim_per_bin']))])
        lines.append(f"Sample {idx:03d} | time={r['time']:.1f} | event={r['event']} | true_bin={r['true_bin']} | best_bin={r['best_bin']} | {sim_str}")

    text = "\n".join(lines)
    print(text)
    if save_path is not None:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[Inspect z0-text] saved to {save_path}")
    return all_sample_stats


def plot_text_condition_similarity(bin0_path, bin1_path, device, num_bins=4, save_path="text_condition_similarity.png"):
    d0 = torch.load(bin0_path, map_location=device)
    d1 = torch.load(bin1_path, map_location=device)
    v_high = F.normalize(d0["text_features"].float(), p=2, dim=1)
    v_low = F.normalize(d1["text_features"].float(), p=2, dim=1)

    conds = []
    for i in range(num_bins):
        alpha = i / (num_bins - 1) if num_bins > 1 else 0.0
        v = (1 - alpha) * v_high + alpha * v_low
        v = F.normalize(v, p=2, dim=1)
        conds.append(v.reshape(-1))

    conds = torch.stack(conds, dim=0)
    conds = F.normalize(conds, p=2, dim=1)
    sim = (conds @ conds.t()).cpu().numpy()

    plt.figure(figsize=(5, 4), dpi=200)
    plt.imshow(sim, cmap="viridis", vmin=0.0, vmax=1.0)
    plt.colorbar()
    plt.xticks(range(num_bins), [f"bin{i}" for i in range(num_bins)])
    plt.yticks(range(num_bins), [f"bin{i}" for i in range(num_bins)])
    plt.title("Interpolated Text Condition Similarity")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Plot] saved to {save_path}")
