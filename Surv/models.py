import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class IntraTissueGatedAttention(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_tissues=6, topk=50, num_bins=4):
        super().__init__()
        self.num_tissues = num_tissues
        self.topk = topk
        self.input_dim = input_dim

        self.attention_V = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Sigmoid())
        self.attention_weights = nn.Linear(hidden_dim, 1)
        self.norm = nn.LayerNorm(input_dim)

        self.compress = nn.Sequential(
            nn.Linear(num_tissues * input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.risk_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_bins),
        )

    def forward(self, x, mask=None):
        b, n, k, d = x.shape
        x_flat = x.view(b * n, k, d)

        a_v = self.attention_V(x_flat)
        a_u = self.attention_U(x_flat)
        scores = self.attention_weights(a_v * a_u).squeeze(-1)

        if mask is not None:
            mask_flat = mask.view(b * n, k)
            scores = scores.masked_fill(mask_flat == 0, -1e9)

        alpha = torch.softmax(scores, dim=1).unsqueeze(-1)
        tissue_tokens = torch.sum(x_flat * alpha, dim=1).view(b, n, d)

        if n != self.num_tissues:
            if n > self.num_tissues:
                tissue_tokens = tissue_tokens[:, : self.num_tissues, :]
            else:
                pad = torch.zeros(b, self.num_tissues - n, d, device=tissue_tokens.device, dtype=tissue_tokens.dtype)
                tissue_tokens = torch.cat([tissue_tokens, pad], dim=1)

        z_0 = self.norm(tissue_tokens)
        z_flat = z_0.reshape(b, -1)
        compressed_feat = self.compress(z_flat)
        logits = self.risk_head(compressed_feat)
        hazards = torch.sigmoid(logits)
        return z_0, hazards


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class LearnableConditionEmbedding(nn.Module):
    def __init__(self, num_bins, num_tissues, text_dim, v_high, v_low):
        super().__init__()
        self.num_bins = num_bins
        self.num_tissues = num_tissues
        self.text_dim = text_dim
        self.K = num_bins

        init_embeds = []
        for i in range(num_bins):
            alpha = i / (num_bins - 1) if num_bins > 1 else 0
            v_mixed = (1 - alpha) * v_high + alpha * v_low
            v_mixed = F.normalize(v_mixed, p=2, dim=-1)
            init_embeds.append(v_mixed)

        init_embeds = torch.stack(init_embeds, dim=0)
        self.register_buffer("embeds", init_embeds)

    def get_all_conditions(self, batch_size):
        normed = F.normalize(self.embeds, p=2, dim=-1)
        return normed.unsqueeze(0).expand(batch_size, -1, -1, -1)

    def get_cond_for_bin(self, bin_idx, batch_size):
        normed = F.normalize(self.embeds, p=2, dim=-1)
        return normed[bin_idx].unsqueeze(0).expand(batch_size, -1, -1)

    def separation_loss(self):
        return torch.tensor(0.0, device=self.embeds.device)


class AdaGN1d(nn.Module):
    def __init__(self, channels, time_dim):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, channels)
        self.time_proj = nn.Linear(time_dim, channels * 2)
        nn.init.zeros_(self.time_proj.weight)
        nn.init.zeros_(self.time_proj.bias)

    def forward(self, x, t_emb):
        normalized = self.group_norm(x)
        style = self.time_proj(t_emb).unsqueeze(-1)
        scale, shift = style.chunk(2, dim=1)
        return normalized * (1 + scale) + shift


class CondAdaGN1d(nn.Module):
    def __init__(self, channels, time_dim, cond_dim):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, channels)
        self.time_proj = nn.Linear(time_dim, channels * 2)
        self.cond_proj = nn.Linear(cond_dim, channels * 2)
        nn.init.zeros_(self.time_proj.weight)
        nn.init.zeros_(self.time_proj.bias)
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

    def forward(self, x, t_emb, c_emb):
        normalized = self.group_norm(x)
        t_style = self.time_proj(t_emb).unsqueeze(-1)
        t_scale, t_shift = t_style.chunk(2, dim=1)
        c_style = self.cond_proj(c_emb).unsqueeze(-1)
        c_scale, c_shift = c_style.chunk(2, dim=1)
        h = normalized * (1 + t_scale) + t_shift
        h = h * (1 + c_scale) + c_shift
        return h


class ResBlock1d_Pointwise(nn.Module):
    def __init__(self, in_c, out_c, time_dim, cond_dim):
        super().__init__()
        self.adagn1 = CondAdaGN1d(in_c, time_dim, cond_dim)
        self.conv1 = nn.Conv1d(in_c, out_c, kernel_size=1)
        self.adagn2 = CondAdaGN1d(out_c, time_dim, cond_dim)
        self.conv2 = nn.Conv1d(out_c, out_c, kernel_size=1)
        self.act = nn.SiLU()
        self.shortcut = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
        nn.init.zeros_(self.conv2.weight)

    def forward(self, x, t_emb, c_emb):
        res = self.shortcut(x)
        h = self.conv1(self.act(self.adagn1(x, t_emb, c_emb)))
        h = self.conv2(self.act(self.adagn2(h, t_emb, c_emb)))
        return h + res


class TissueDiffusionModel(nn.Module):
    def __init__(self, x_dim=512, text_dim=512, time_dim=256, cond_dim=128, num_layers=2):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(text_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.input_proj = nn.Conv1d(x_dim + text_dim, x_dim, 1)
        self.blocks = nn.ModuleList([ResBlock1d_Pointwise(x_dim, x_dim, time_dim, cond_dim) for _ in range(num_layers)])
        self.final_adagn = CondAdaGN1d(x_dim, time_dim, cond_dim)
        self.output_conv = nn.Conv1d(x_dim, x_dim, 1)
        nn.init.zeros_(self.output_conv.weight)

    def forward_denoise(self, z_t, t, text_cond_seq):
        x_in = torch.cat([z_t, text_cond_seq], dim=-1)
        x = x_in.permute(0, 2, 1)
        t_emb = self.time_mlp(t)
        c_emb = self.cond_mlp(text_cond_seq.mean(dim=1))
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x, t_emb, c_emb)
        x = self.output_conv(F.silu(self.final_adagn(x, t_emb, c_emb)))
        return x.permute(0, 2, 1)


class HazardCalibrator(nn.Module):
    def __init__(self, num_bins: int, init_scale: float = 1.0, init_bias: float = 0.0):
        super().__init__()
        self.num_bins = int(num_bins)
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.bias = nn.Parameter(torch.full((self.num_bins,), float(init_bias)))

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        logits = scores * self.scale + self.bias
        return torch.sigmoid(logits)
