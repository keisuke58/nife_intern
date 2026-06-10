#!/usr/bin/env python3
"""
benchmark_flow_matching.py — ①' Conditional Flow Matching for LOO trajectory prediction.

Flow Matching (Lipman et al. 2023 / rectified flow, Liu et al. 2023) is the
ODE-based successor to diffusion. Instead of learning a denoiser over a
stochastic reverse SDE, we learn a *velocity field* v(x,t) and integrate a
deterministic ODE  dx/dt = v(x,t)  from noise (t=0) to data (t=1).

Why this is native to nife: the whole repo is ODE-based (gLV/replicator).
Flow Matching is conceptually the same object — a learned ODE — so it is a
principled sibling of the mechanistic model, not an alien generative trick.

Conditional rectified flow:
  x0 ~ N(0, I)                 (base / noise)
  x1 = phi(t+1)                (target composition)
  x_t = (1-t) x0 + t x1        (linear interpolant)
  target velocity u = x1 - x0
  loss: || v_theta(x_t, t, cond=phi(t)) - u ||^2
  inference: integrate dx/dt = v from x0~N(0,I), t:0->1 (few Euler steps)

Same LOO-CV protocol as benchmark_diffusion.py for a head-to-head row.

Usage (repo root):
  python scripts/analysis/benchmark_flow_matching.py [--epochs 2000] [--steps 20]

Output:
  results/benchmark_baselines/fm_rmse_per_fold.json
  results/benchmark_baselines/rmse_table.json  (adds 'Flow Matching (conditional)')
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


# ── Velocity network ──────────────────────────────────────────────────────────

class VelocityNet(nn.Module):
    """v(x_t, t, cond): predicts the flow velocity (same dim as x)."""

    def __init__(self, n_g=N_G, hidden=128, t_dim=32):
        super().__init__()
        self.t_emb = nn.Sequential(
            nn.Linear(1, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(n_g + n_g + t_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, n_g),
        )

    def forward(self, x_t, t_norm, cond):
        te = self.t_emb(t_norm.unsqueeze(-1))
        return self.net(torch.cat([cond, x_t, te], dim=-1))


# ── Conditional rectified flow ────────────────────────────────────────────────

class CondFlowMatching:
    def __init__(self, hidden=128, device='cpu'):
        self.device = device
        self.net    = VelocityNet(hidden=hidden).to(device)

    def train(self, X_tr, y_tr, epochs=2000, lr=3e-4, batch=18):
        X = torch.tensor(X_tr, dtype=torch.float32).to(self.device)
        y = torch.tensor(y_tr, dtype=torch.float32).to(self.device)
        N = X.shape[0]
        opt = optim.Adam(self.net.parameters(), lr=lr)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        self.net.train()
        for ep in range(epochs):
            idx  = torch.randint(0, N, (min(batch, N),))
            x1   = y[idx]                                  # target
            cond = X[idx]                                  # phi(t)
            x0   = torch.randn_like(x1)                    # base noise
            t    = torch.rand(len(idx), 1).to(self.device) # uniform t in [0,1]
            x_t  = (1 - t) * x0 + t * x1                    # linear path
            u    = x1 - x0                                  # target velocity
            v    = self.net(x_t, t.squeeze(-1), cond)
            loss = nn.functional.mse_loss(v, u)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()

    @torch.no_grad()
    def sample(self, phi_cond, n_steps=20):
        """Integrate dx/dt = v from x0~N(0,I), t:0->1 with Euler steps."""
        self.net.eval()
        cond = torch.tensor(phi_cond, dtype=torch.float32).to(self.device).unsqueeze(0)
        x  = torch.randn(1, N_G).to(self.device)
        dt = 1.0 / n_steps
        for k in range(n_steps):
            t  = torch.full((1,), k * dt).to(self.device)
            v  = self.net(x, t, cond)
            x  = x + dt * v
        out = x.squeeze(0).cpu().numpy()
        out = np.clip(out, 0, None)
        s = out.sum()
        return out / s if s > 1e-12 else np.ones(N_G) / N_G


# ── LOO-CV ────────────────────────────────────────────────────────────────────

def loo_rmse_fm(phi_all, hold, epochs=2000, n_steps=20, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    tr_idx = [i for i in range(N_FOLDS) if i != hold]
    phi_tr = phi_all[tr_idx]
    X_tr   = phi_tr[:, :-1, :].reshape(-1, N_G)
    y_tr   = phi_tr[:, 1:,  :].reshape(-1, N_G)
    phi_ho = phi_all[hold]

    model = CondFlowMatching()
    model.train(X_tr, y_tr, epochs=epochs)

    pred_W2 = model.sample(phi_ho[0], n_steps=n_steps)
    pred_W3 = model.sample(pred_W2,   n_steps=n_steps)

    mse_W2 = np.mean((pred_W2 - phi_ho[1])**2)
    mse_W3 = np.mean((pred_W3 - phi_ho[2])**2)
    return float(np.sqrt((mse_W2 + mse_W3) / 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--steps',  type=int, default=20, help='ODE Euler steps at inference')
    parser.add_argument('--seed',   type=int, default=42)
    args = parser.parse_args()

    phi_all = np.load(PHI_NPY)
    print(f'phi: {phi_all.shape}  epochs={args.epochs}  ODE-steps={args.steps}', flush=True)
    print('Conditional Flow Matching (rectified flow) — ODE-based diffusion successor', flush=True)

    rmses = []
    for fold in range(N_FOLDS):
        r = loo_rmse_fm(phi_all, fold, epochs=args.epochs, n_steps=args.steps, seed=args.seed)
        rmses.append(r)
        print(f'  FM  fold {fold} ({PATIENTS[fold]})  RMSE={r:.4f}', flush=True)

    mean_r, std_r = float(np.mean(rmses)), float(np.std(rmses))
    print(f'\n  Flow Matching  MEAN={mean_r:.4f}  STD={std_r:.4f}', flush=True)

    out = {'fm_per_fold': rmses, 'mean': mean_r, 'std': std_r,
           'epochs': args.epochs, 'ode_steps': args.steps}
    json.dump(out, open(OUT_DIR / 'fm_rmse_per_fold.json', 'w'), indent=2)

    table_path = OUT_DIR / 'rmse_table.json'
    if table_path.exists():
        table = json.load(open(table_path))
        table['Flow Matching (conditional)'] = {
            'mean_rmse': mean_r, 'std_rmse': std_r, 'per_fold': rmses}
        json.dump(table, open(table_path, 'w'), indent=2)
        print(f'Updated: {table_path}')

    print(f'Saved: {OUT_DIR / "fm_rmse_per_fold.json"}')


if __name__ == '__main__':
    main()
