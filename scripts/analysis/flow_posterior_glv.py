#!/usr/bin/env python3
"""
flow_posterior_glv.py — ②' Probability-flow ODE posterior over the gLV A matrix.

This is the deterministic-ODE counterpart of the score-based Langevin posterior
(score_posterior_glv.py). It demonstrates the SDE -> ODE migration that mirrors
nife's own ODE foundation:

    Langevin (SDE):  dA = score(A) dt + sqrt(2) dW      (stochastic, slow mixing)
    Flow Matching (ODE):  dA/dt = v_theta(A, t)         (deterministic, fast)

Construction (amortized posterior via rectified flow):
  1. Take the Langevin posterior samples A_samples.npy as ground-truth draws
     from p(A | phi_obs)  (the expensive SDE already ran once).
  2. Train an unconditional Flow Matching model transporting a broad Gaussian
     base -> those posterior samples.
  3. Resample the posterior in a few deterministic ODE steps (probability flow),
     thousands of times faster than re-running Langevin.

Payoff:
  - amortization: new posterior draws are ~free after one flow fit
  - the flow gives a smooth, differentiable posterior over A
  - mean/std/sign agreement vs Langevin validates the flow is faithful

Usage (repo root, run score_posterior_glv.py first):
  python scripts/analysis/flow_posterior_glv.py [--epochs 4000] [--n-gen 2000]

Output:
  results/score_posterior/flow_A_samples.npy
  results/score_posterior/flow_summary.json
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

import argparse, json
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from guild_replicator_dieckow import N_G

_here = Path(__file__).resolve().parents[2]
SP_DIR  = _here / 'results' / 'score_posterior'
DIM     = N_G * N_G   # 100 flattened A entries


# ── Velocity network (unconditional) ──────────────────────────────────────────

class AVelocityNet(nn.Module):
    def __init__(self, dim=DIM, hidden=256, t_dim=64):
        super().__init__()
        self.t_emb = nn.Sequential(
            nn.Linear(1, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(dim + t_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x_t, t_norm):
        te = self.t_emb(t_norm.unsqueeze(-1))
        return self.net(torch.cat([x_t, te], dim=-1))


# ── Rectified flow on posterior samples ───────────────────────────────────────

class FlowPosterior:
    def __init__(self, hidden=256, device='cpu'):
        self.device = device
        self.net    = AVelocityNet(hidden=hidden).to(device)

    def fit(self, A_post, epochs=4000, lr=3e-4, batch=64):
        """A_post: (N, 100) Langevin posterior samples (standardised internally)."""
        self.mean = A_post.mean(0)
        self.std  = A_post.std(0) + 1e-8
        Z = (A_post - self.mean) / self.std         # standardise target
        Zt = torch.tensor(Z, dtype=torch.float32).to(self.device)
        N  = Zt.shape[0]
        opt = optim.Adam(self.net.parameters(), lr=lr)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        self.net.train()
        for ep in range(epochs):
            idx = torch.randint(0, N, (min(batch, N),))
            x1  = Zt[idx]                              # posterior target
            x0  = torch.randn_like(x1)                 # Gaussian base
            t   = torch.rand(len(idx), 1).to(self.device)
            x_t = (1 - t) * x0 + t * x1
            u   = x1 - x0
            v   = self.net(x_t, t.squeeze(-1))
            loss = nn.functional.mse_loss(v, u)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        return float(loss.item())

    @torch.no_grad()
    def sample(self, n, n_steps=50):
        """Probability-flow ODE: integrate dx/dt = v from N(0,I), t:0->1."""
        self.net.eval()
        x  = torch.randn(n, DIM).to(self.device)
        dt = 1.0 / n_steps
        for k in range(n_steps):
            t = torch.full((n,), k * dt).to(self.device)
            x = x + dt * self.net(x, t)
        Z = x.cpu().numpy()
        return Z * self.std + self.mean             # de-standardise


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=4000)
    parser.add_argument('--n-gen',  type=int, default=2000, help='flow-generated samples')
    parser.add_argument('--ode-steps', type=int, default=50)
    parser.add_argument('--seed',   type=int, default=0)
    args = parser.parse_args()

    samples_path = SP_DIR / 'A_samples.npy'
    if not samples_path.exists():
        print(f'Missing {samples_path}')
        print('Run: python scripts/analysis/score_posterior_glv.py first')
        return

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    A_lang = np.load(samples_path)                 # (n_lang, 100)
    burnin = len(A_lang) // 2
    A_post = A_lang[burnin:]                        # post-burnin Langevin draws
    print(f'Langevin posterior samples: {A_post.shape}')

    flow = FlowPosterior()
    print(f'Training rectified flow ({args.epochs} epochs)...', flush=True)
    final_loss = flow.fit(A_post, epochs=args.epochs)
    print(f'  final FM loss: {final_loss:.5f}')

    A_flow = flow.sample(args.n_gen, n_steps=args.ode_steps)   # (n_gen, 100)
    np.save(SP_DIR / 'flow_A_samples.npy', A_flow)
    print(f'Flow-generated samples: {A_flow.shape}')

    # ── Fidelity: compare Langevin vs Flow posterior moments ──────────────────
    m_lang, s_lang = A_post.mean(0), A_post.std(0)
    m_flow, s_flow = A_flow.mean(0), A_flow.std(0)

    mean_corr = float(np.corrcoef(m_lang, m_flow)[0, 1])
    std_corr  = float(np.corrcoef(s_lang, s_flow)[0, 1])
    sign_agree = float((np.sign(m_lang) == np.sign(m_flow)).mean())
    mean_mae   = float(np.abs(m_lang - m_flow).mean())

    print('\n=== Flow posterior fidelity (vs Langevin) ===')
    print(f'  posterior-mean correlation : {mean_corr:.4f}')
    print(f'  posterior-std  correlation : {std_corr:.4f}')
    print(f'  sign agreement (100 entries): {sign_agree:.1%}')
    print(f'  mean MAE                    : {mean_mae:.5f}')

    summary = {
        'n_langevin_post': int(len(A_post)),
        'n_flow_generated': int(args.n_gen),
        'epochs': args.epochs,
        'ode_steps': args.ode_steps,
        'final_fm_loss': final_loss,
        'mean_corr': mean_corr,
        'std_corr': std_corr,
        'sign_agreement': sign_agree,
        'mean_mae': mean_mae,
        'A_mean_flow': m_flow.tolist(),
        'A_std_flow':  s_flow.tolist(),
    }
    json.dump(summary, open(SP_DIR / 'flow_summary.json', 'w'), indent=2)
    print(f'\nSaved: {SP_DIR / "flow_A_samples.npy"}')
    print(f'Saved: {SP_DIR / "flow_summary.json"}')


if __name__ == '__main__':
    main()
