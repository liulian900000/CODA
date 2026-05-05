import torch


def get_alphas_cumprod(num_timesteps=1000, device='cuda'):
    scale = 1000 / num_timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
    alphas = 1 - betas
    return torch.cumprod(alphas, dim=0)


def q_sample(z_0, t, noise, alphas_cumprod):
    alpha_bar = alphas_cumprod[t].view(-1, 1, 1)
    return torch.sqrt(alpha_bar) * z_0 + torch.sqrt(1 - alpha_bar) * noise
