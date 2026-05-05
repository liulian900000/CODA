import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class IntraTissueGatedAttention(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_tissues=3, topk=50):
        super().__init__()
        self.num_tissues = num_tissues
        self.topk = topk
        self.input_dim = input_dim

        self.attention_V = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Sigmoid())
        self.attention_weights = nn.Linear(hidden_dim, 1)
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(num_tissues * input_dim, 2)

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
        return self.norm(tissue_tokens)

    def get_logits(self, z_0):
        b = z_0.shape[0]
        z_flat = z_0.view(b, -1)
        return self.classifier(self.dropout(z_flat))


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


class AdaGN1d(nn.Module):
    def __init__(self, channels, time_dim, text_dim):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, channels)
        self.time_proj = nn.Linear(time_dim, channels * 2)
        self.text_proj = nn.Linear(text_dim, channels * 2)
        nn.init.zeros_(self.time_proj.weight)
        nn.init.zeros_(self.time_proj.bias)
        nn.init.zeros_(self.text_proj.weight)
        nn.init.zeros_(self.text_proj.bias)

    def forward(self, x, t_emb, text_emb):
        normalized = self.group_norm(x)
        style_t = self.time_proj(t_emb).unsqueeze(-1)
        style_txt = self.text_proj(text_emb).permute(0, 2, 1)
        scale, shift = (style_t + style_txt).chunk(2, dim=1)
        return normalized * (1 + scale) + shift


class ResBlock1d_Pointwise(nn.Module):
    def __init__(self, in_c, out_c, time_dim, text_dim):
        super().__init__()
        self.adagn1 = AdaGN1d(in_c, time_dim, text_dim)
        self.conv1 = nn.Conv1d(in_c, out_c, kernel_size=1)
        self.adagn2 = AdaGN1d(out_c, time_dim, text_dim)
        self.conv2 = nn.Conv1d(out_c, out_c, kernel_size=1)
        self.act = nn.SiLU()
        self.shortcut = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
        nn.init.zeros_(self.conv2.weight)

    def forward(self, x, t_emb, txt_emb):
        res = self.shortcut(x)
        h = self.conv1(self.act(self.adagn1(x, t_emb, txt_emb)))
        h = self.conv2(self.act(self.adagn2(h, t_emb, txt_emb)))
        return h + res


class TissueDiffusionModel(nn.Module):
    def __init__(self, x_dim=512, text_dim=512, time_dim=256, num_layers=6):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.input_proj = nn.Conv1d(x_dim, x_dim, 1)
        self.blocks = nn.ModuleList([ResBlock1d_Pointwise(x_dim, x_dim, time_dim, text_dim) for _ in range(num_layers)])
        self.final_adagn = AdaGN1d(x_dim, time_dim, text_dim)
        self.output_conv = nn.Conv1d(x_dim, x_dim, 1)
        nn.init.zeros_(self.output_conv.weight)

    def forward_denoise(self, z_t, t, text_cond_seq):
        if self.training:
            z_t = F.dropout(z_t, p=0.1)
        x = z_t.permute(0, 2, 1)
        t_emb = self.time_mlp(t)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x, t_emb, text_cond_seq)
        x = self.output_conv(F.silu(self.final_adagn(x, t_emb, text_cond_seq)))
        return x.permute(0, 2, 1)
