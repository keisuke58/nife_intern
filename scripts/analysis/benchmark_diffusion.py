#!/usr/bin/env python3
"""
benchmark_diffusion.py — ① Conditional DDPM for LOO trajectory prediction.

Adds a score-based diffusion model to the biology AI benchmark alongside
Ridge/LASSO/RF baselines. With n=18 training pairs the network intentionally
underfits — that IS the finding: gLV mechanistic structure beats data-hungry
generative models under the small-data regime of oral microbiome studies.

Architecture:
  Conditioned DDPM (Ho et al. 2020, simplified)
  Score net: MLP(phi_cond, phi_noisy, t_embed) -> noise_pred
  T = 500 diffusion steps, cosine schedule
  Inference: DDPM reverse (deterministic DDIM-style)

Usage (repo root):
  python scripts/analysis/benchmark_diffusion.py [--epochs 2000] [--seed 42]

Output:
  results/benchmark_baselines/ddpm_rmse_per_fold.json
  results/benchmark_baselines/rmse_table.json  (updated with DDPM column)
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

import argparse, json
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

_here = Path(__file__).resolve().parents[2]
PHI_NPY  = _here / 'results' / 'dieckow_otu' / 'phi_guild.npy'
OUT_DIR  = _here / 'results' / 'benchmark_baselines'
OUT_DIR.mkdir(exist_ok=True)

N_G     = 10
N_FOLDS = 10
PATIENTS = list('ABCDEFGHKL')


# ── Cosine noise schedule ─────────────────────────────────────────────────────

def cosine_schedule(T, s=0.008):
    t = torch.arange(T + 1, dtype=torch.float32)
    f = torch.cos(((t / T + s) / (1 + s)) * (torch.pi / 2)) ** 2
    alphas_cumprod = f / f[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clamp(betas, 0.0001, 0.9999), alphas_cumprod[1:]


# ── Score network ─────────────────────────────────────────────────────────────

class ScoreNet(nn.Module):
    """MLP conditioned on phi(W1) + noisy phi(t) + timestep embedding."""

    def __init__(self, n_g=N_G, hidden=128, t_dim=32):
        super().__init__()
        self.t_emb = nn.Sequential(
            nn.Linear(1, t_dim), nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(n_g + n_g + t_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, n_g),
        )

    def forward(self, phi_cond, phi_noisy, t_norm):
        # t_norm in [0, 1]
        te = self.t_emb(t_norm.unsqueeze(-1))
        x  = torch.cat([phi_cond, phi_noisy, te], dim=-1)
        return self.net(x)


# ── DDPM training + inference ─────────────────────────────────────────────────

class CondDDPM:
    def __init__(self, T=500, hidden=128, device='cpu'):
        self.T      = T
        self.device = device
        self.net    = ScoreNet(hidden=hidden).to(device)
        betas, alphas_cp = cosine_schedule(T)
        self.betas       = betas.to(device)
        self.alphas_cp   = alphas_cp.to(device)
        self.sqrt_acp    = torch.sqrt(self.alphas_cp)
        self.sqrt_1macp  = torch.sqrt(1 - self.alphas_cp)

    def _add_noise(self, x0, t_idx):
        eps  = torch.randn_like(x0)
        xt   = (self.sqrt_acp[t_idx, None] * x0 +
                self.sqrt_1macp[t_idx, None] * eps)
        return xt, eps

    def train(self, X_tr, y_tr, epochs=2000, lr=3e-4, batch=18):
        """X_tr: (N,10) phi(t), y_tr: (N,10) phi(t+1)."""
        X  = torch.tensor(X_tr, dtype=torch.float32).to(self.device)
        y  = torch.tensor(y_tr, dtype=torch.float32).to(self.device)
        N  = X.shape[0]
        opt = optim.Adam(self.net.parameters(), lr=lr)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        self.net.train()
        for ep in range(epochs):
            idx = torch.randint(0, N, (min(batch, N),))
            x0  = y[idx]
            cond = X[idx]
            t   = torch.randint(0, self.T, (len(idx),)).to(self.device)
            xt, eps = self._add_noise(x0, t)
            t_norm  = (t.float() / self.T).to(self.device)
            eps_pred = self.net(cond, xt, t_norm)
            loss = nn.functional.mse_loss(eps_pred, eps)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()

    @torch.no_grad()
    def sample(self, phi_cond):
        """DDIM deterministic reverse from pure noise, conditioned on phi_cond."""
        self.net.eval()
        cond = torch.tensor(phi_cond, dtype=torch.float32).to(self.device).unsqueeze(0)
        xt   = torch.randn(1, N_G).to(self.device)
        for t_idx in reversed(range(self.T)):
            t_norm = torch.tensor([t_idx / self.T]).to(self.device)
            eps_pred = self.net(cond, xt, t_norm)
            acp_t = self.alphas_cp[t_idx]
            acp_t_prev = self.alphas_cp[t_idx - 1] if t_idx > 0 else torch.tensor(1.0).to(self.device)
            # DDIM deterministic step
            x0_pred = (xt - torch.sqrt(1 - acp_t) * eps_pred) / torch.sqrt(acp_t)
            x0_pred = torch.clamp(x0_pred, 0, None)
            xt = torch.sqrt(acp_t_prev) * x0_pred + torch.sqrt(1 - acp_t_prev) * eps_pred
        phi_out = xt.squeeze(0).cpu().numpy()
        phi_out = np.clip(phi_out, 0, None)
        s = phi_out.sum()
        return phi_out / s if s > 1e-12 else np.ones(N_G) / N_G


# ── LOO-CV ────────────────────────────────────────────────────────────────────

def loo_rmse_ddpm(phi_all, hold, epochs=2000, T=500, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr_idx = [i for i in range(N_FOLDS) if i != hold]
    phi_tr = phi_all[tr_idx]                          # (9,3,10)
    X_tr   = phi_tr[:, :-1, :].reshape(-1, N_G)       # (18,10)
    y_tr   = phi_tr[:, 1:,  :].reshape(-1, N_G)       # (18,10)
    phi_ho = phi_all[hold]                            # (3,10)

    model = CondDDPM(T=T)
    model.train(X_tr, y_tr, epochs=epochs)

    phi_pred_W2 = model.sample(phi_ho[0])
    phi_pred_W3 = model.sample(phi_pred_W2)

    mse_W2 = np.mean((phi_pred_W2 - phi_ho[1])**2)
    mse_W3 = np.mean((phi_pred_W3 - phi_ho[2])**2)
    return float(np.sqrt((mse_W2 + mse_W3) / 2))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--T',      type=int, default=500,  help='diffusion steps')
    parser.add_argument('--seed',   type=int, default=42)
    args = parser.parse_args()

    phi_all = np.load(PHI_NPY)
    print(f'phi: {phi_all.shape}  epochs={args.epochs}  T={args.T}')

    rmses = []
    for fold in range(N_FOLDS):
        r = loo_rmse_ddpm(phi_all, fold, epochs=args.epochs, T=args.T, seed=args.seed)
        rmses.append(r)
        print(f'  DDPM  fold {fold} ({PATIENTS[fold]})  RMSE={r:.4f}')

    mean_r, std_r = float(np.mean(rmses)), float(np.std(rmses))
    print(f'\n  DDPM  MEAN={mean_r:.4f}  STD={std_r:.4f}')

    # Save per-fold
    out = {'ddpm_per_fold': rmses, 'mean': mean_r, 'std': std_r,
           'epochs': args.epochs, 'T': args.T}
    json.dump(out, open(OUT_DIR / 'ddpm_rmse_per_fold.json', 'w'), indent=2)

    # Merge into rmse_table.json
    table_path = OUT_DIR / 'rmse_table.json'
    if table_path.exists():
        table = json.load(open(table_path))
        table['DDPM (conditional)'] = {'mean_rmse': mean_r, 'std_rmse': std_r,
                                        'per_fold': rmses}
        json.dump(table, open(table_path, 'w'), indent=2)
        print(f'Updated: {table_path}')

    print(f'Saved: {OUT_DIR / "ddpm_rmse_per_fold.json"}')


if __name__ == '__main__':
    main()
