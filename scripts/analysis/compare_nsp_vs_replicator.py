#!/usr/bin/env python3
"""
compare_nsp_vs_replicator.py

NSP full Hamilton ODE (φ,ψ,φ0,γ — implicit Euler+Newton) vs
Replicator ODE (φ only — RK45) の数値比較。

同じ A 行列・初期条件で両モデルを走らせ、軌道差を定量する。
条件:
  - ψ_i = 1 固定 (NSP Phase-1 と等価)
  - α* = 0 (抗生物質なし)
  - c* = 25 (NSP デフォルト)
  - n_sp = 5, n_steps = 2500, dt = 1e-4
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root: bare sibling imports

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pathlib import Path

# ── パス設定 ────────────────────────────────────────────────────────────
sys.path.insert(0, '/home/nishioka/IKM_Hiwi/data_5species/main')
from hamilton_ode_jax_nsp import simulate_0d_nsp, theta_to_matrices, count_params

SPECIES = ['So', 'An', 'Vei', 'Fn', 'Pg']
N_SP    = 5
N_STEPS = 2500
DT      = 1e-4
C_STAR  = 25.0

# ── テスト用 A 行列 (DH MAP に近い典型値) ────────────────────────────────
# 対称行列 — NSP 論文の DH 条件の MAP に近い値
A_mat = np.array([
    [ 1.2,  0.8, -1.5,  0.3, -0.5],
    [ 0.8,  1.0,  0.5,  0.2, -0.1],
    [-1.5,  0.5,  2.0,  1.2,  1.8],
    [ 0.3,  0.2,  1.2,  4.4,  5.6],
    [-0.5, -0.1,  1.8,  5.6,  3.4],
], dtype=np.float64)

# 初期条件: So, An 優位 (commensal-like)
phi0_sp = np.array([0.40, 0.30, 0.15, 0.10, 0.05], dtype=np.float64)

# ── theta に変換 (upper triangle, column-major) ─────────────────────────
def A_to_upper_triangle(A):
    n = A.shape[0]
    entries = []
    for j in range(n):
        for i in range(j + 1):
            entries.append(A[i, j])
    return np.array(entries)

theta_A = A_to_upper_triangle(A_mat)       # 15 params
theta_b = np.zeros(N_SP, dtype=np.float64) # b=0 (Dieckow では b は外でパラメータ化)
# append K_hill=0.0 so simulate_0d_nsp can index theta[n_A+n_sp] safely
theta   = np.concatenate([theta_A, theta_b, [0.0]])

print("=" * 60)
print(f"A matrix:\n{A_mat}")
print(f"phi0: {phi0_sp}")
print(f"theta (len={len(theta)}): {theta}")

# ── Model 1: NSP (φ,ψ,φ0,γ — implicit Euler+Newton via JAX) ────────────
print("\n[1] Running NSP model (implicit Euler+Newton)...")
import jax
jax.config.update('jax_enable_x64', True)

# simulate_0d_nsp returns shape (n_steps+1, n_sp) — phi only (after extracting)
# psi_fixed = 1 corresponds to not passing psi_obs (Phase-1 with ψ=1 init)
import time

t0 = time.time()
phi_nsp_raw = simulate_0d_nsp(
    theta,
    n_sp=N_SP,
    n_steps=N_STEPS,
    dt=DT,
    phi_init=phi0_sp,
    c_const=C_STAR,
    alpha_const=0.0,
    K_hill=0.0,
    n_hill=4.0,
)
phi_nsp = np.array(phi_nsp_raw)  # (n_steps+1, n_sp)
t_nsp = time.time() - t0
print(f"  Done in {t_nsp:.2f}s, shape={phi_nsp.shape}")
print(f"  Final phi (NSP): {phi_nsp[-1]}")

# ── Model 2: Replicator (RK45 via scipy) ─────────────────────────────────
print("\n[2] Running Replicator (RK45)...")

def replicator_rhs(t, phi, A, b):
    f = b + A @ phi
    fbar = phi @ f
    return phi * (f - fbar)

t_span = [0.0, N_STEPS * DT]
t_eval = np.linspace(0, N_STEPS * DT, N_STEPS + 1)

t0 = time.time()
sol = solve_ivp(
    replicator_rhs,
    t_span,
    phi0_sp,
    args=(A_mat, theta_b),
    method='RK45',
    t_eval=t_eval,
    rtol=1e-8,
    atol=1e-10,
)
t_rep = time.time() - t0
phi_rep = sol.y.T   # (n_steps+1, n_sp)
print(f"  Done in {t_rep:.2f}s, shape={phi_rep.shape}")
print(f"  Final phi (Replicator): {phi_rep[-1]}")

# ── 比較計算 ──────────────────────────────────────────────────────────────
print("\n[3] Computing differences...")

# NSPをreplicatorと同じ時点に対応させる
phi_nsp_end = phi_nsp[-1]
phi_rep_end = phi_rep[-1]

# 正規化 (NSP は φ₀ 込みなので species だけで simplex に戻す)
phi_nsp_norm = phi_nsp / (phi_nsp.sum(axis=1, keepdims=True) + 1e-15)

diff = np.abs(phi_nsp_norm - phi_rep)   # (n_steps+1, n_sp)
rmse_total = np.sqrt(np.mean(diff**2))
max_diff   = diff.max()
bc_final   = 0.5 * np.sum(np.abs(phi_nsp_norm[-1] - phi_rep[-1]))

print(f"\n  RMSE (全時点): {rmse_total:.6f}")
print(f"  最大差 (全時点): {max_diff:.6f}")
print(f"  Bray-Curtis 最終時点: {bc_final:.6f}")
print(f"\n  最終値比較:")
print(f"  {'種':>5}  {'NSP':>8}  {'Replctr':>8}  {'|差|':>8}")
for i, sp in enumerate(SPECIES):
    print(f"  {sp:>5}  {phi_nsp_norm[-1, i]:8.4f}  {phi_rep[-1, i]:8.4f}  {abs(phi_nsp_norm[-1,i]-phi_rep[-1,i]):8.4f}")

# 時系列での RMSE (種別)
print(f"\n  種別 RMSE (全時点):")
for i, sp in enumerate(SPECIES):
    r = np.sqrt(np.mean((phi_nsp_norm[:, i] - phi_rep[:, i])**2))
    print(f"  {sp:>5}: {r:.6f}")

# ── 図: 軌道比較 ──────────────────────────────────────────────────────────
t_arr = t_eval

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']

for i, sp in enumerate(SPECIES):
    ax = axes[i]
    ax.plot(t_arr, phi_rep[:, i], '-', color=colors[i],
            lw=2, label='Replicator (RK45)')
    ax.plot(t_arr, phi_nsp_norm[:, i], '--', color=colors[i],
            lw=2, alpha=0.7, label='NSP (implicit Euler)')
    ax.set_title(sp, fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (dim-less)')
    ax.set_ylabel('φ')
    rmse_i = np.sqrt(np.mean((phi_nsp_norm[:, i] - phi_rep[:, i])**2))
    ax.text(0.05, 0.95, f'RMSE={rmse_i:.4f}',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    if i == 0:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# 差の時系列
ax = axes[5]
for i, sp in enumerate(SPECIES):
    ax.plot(t_arr, diff[:, i], color=colors[i], lw=1.5, label=sp)
ax.set_title('|NSP − Replicator|', fontsize=12, fontweight='bold')
ax.set_xlabel('Time')
ax.set_ylabel('|Δφ|')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.suptitle(
    f'NSP (implicit Euler+Newton) vs Replicator (RK45)\n'
    f'RMSE={rmse_total:.5f}  BC_final={bc_final:.5f}  '
    f'NSP:{t_nsp:.2f}s  Rep:{t_rep:.2f}s',
    fontsize=11
)
plt.tight_layout()
out_fig = Path('/home/nishioka/IKM_Hiwi/nife/figures/nsp_vs_replicator.png')
out_fig.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_fig, dpi=150, bbox_inches='tight')
print(f"\n  Figure saved: {out_fig}")

# ── 追加: 様々な初期条件での平均差 ─────────────────────────────────────
print("\n[4] Monte Carlo: 20 random initial conditions...")
rng = np.random.default_rng(42)
all_rmse, all_bc = [], []
for trial in range(20):
    raw = rng.exponential(1.0, N_SP)
    phi_init_r = raw / raw.sum()

    phi_nsp_r = np.array(simulate_0d_nsp(
        theta, n_sp=N_SP, n_steps=N_STEPS, dt=DT,
        phi_init=phi_init_r, c_const=C_STAR, alpha_const=0.0,
        K_hill=0.0, n_hill=4.0,
    ))
    sol_r = solve_ivp(
        replicator_rhs, t_span, phi_init_r,
        args=(A_mat, theta_b), method='RK45',
        t_eval=t_eval, rtol=1e-8, atol=1e-10,
    )
    phi_rep_r = sol_r.y.T
    phi_nsp_rn = phi_nsp_r / (phi_nsp_r.sum(axis=1, keepdims=True) + 1e-15)
    d = np.abs(phi_nsp_rn - phi_rep_r)
    all_rmse.append(np.sqrt(np.mean(d**2)))
    all_bc.append(0.5 * np.sum(np.abs(phi_nsp_rn[-1] - phi_rep_r[-1])))

print(f"  RMSE  mean={np.mean(all_rmse):.5f}  std={np.std(all_rmse):.5f}  max={np.max(all_rmse):.5f}")
print(f"  BC    mean={np.mean(all_bc):.5f}   std={np.std(all_bc):.5f}   max={np.max(all_bc):.5f}")

print("\n=== DONE ===")
