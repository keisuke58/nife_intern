"""Neural ODE with metabolic sign-prior constraint for oral biofilm dynamics.

Learns interaction matrix A_neural via a linear replicator ODE whose RHS is
parameterized by a 10×10 weight matrix (+ per-patient growth bias).
A sign-consistency penalty from the AGORA/KEGG metabolic prior is added to
the MSE loss, analogous to but more flexible than the hard-constraint gLV fit.

The key difference from the vanilla gLV fit: A_neural is learned jointly
across all patients without assuming fixed functional form for b_i, and
the sign penalty acts as a *differentiable soft constraint* via JAX autograd.

Requires: jax, optax (pip install optax)
Falls back gracefully if jax/optax unavailable.

Usage:
    python scripts/fitting/neural_ode_sign_prior.py [--epochs 2000] [--lr 3e-3]

Outputs:
    results/neural_ode_sign/results.json
    results/neural_ode_sign/fig_neural_ode.pdf
"""
# [nife-pathshim]
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

from guild_replicator_dieckow import GUILD_ORDER, GUILD_SHORT, N_G
from loo_cv_kegg_prior import build_net_flow
from pub_style import apply as apply_pub_style

_here = Path(__file__).resolve().parents[2]
PHI_NPY   = _here / 'results' / 'dieckow_otu' / 'phi_guild.npy'
FIT_FILE  = _here / 'results' / 'dieckow_cr' / 'fit_glv_hamilton_kegg_prior.json'
OUT_DIR   = _here / 'results' / 'neural_ode_sign'
T_STEPS   = np.array([0.0, 1.0, 2.0])  # weeks


def replicator_euler(phi0, A, b, n_steps=50):
    """Fast Euler integration of replicator ODE over 1 unit interval."""
    dt = 1.0 / n_steps
    phi = phi0.copy()
    for _ in range(n_steps):
        f = b + A @ phi
        dphi = phi * (f - phi @ f)
        phi = phi + dt * dphi
        phi = np.clip(phi, 0, None)
        s = phi.sum()
        if s > 1e-12:
            phi = phi / s
    return phi


def sign_penalty_np(A, net_flow, sigma=0.1):
    sp = np.sign(net_flow)
    mask = (sp != 0) & (~np.eye(N_G, dtype=bool))
    pen = 0.0
    for i in range(N_G):
        for j in range(N_G):
            if mask[i, j]:
                w = abs(net_flow[i, j])
                v = max(0.0, -sp[i, j] * A[i, j])
                pen += w * v * v / (2 * sigma ** 2)
    return pen


def run_jax(phi_obs, net_flow, args):
    """Full JAX+optax training."""
    import jax
    import jax.numpy as jnp
    import optax

    n_p, n_t, _ = phi_obs.shape
    phi_jax = jnp.array(phi_obs)
    nf_jax  = jnp.array(net_flow)

    @jax.jit
    def replicator_step(phi, A, b):
        """RK4 over [0,1] — one weekly step."""
        def rhs(phi_):
            f = b + A @ phi_
            return phi_ * (f - phi_ @ f)

        k1 = rhs(phi)
        k2 = rhs(phi + 0.5 * k1)
        k3 = rhs(phi + 0.5 * k2)
        k4 = rhs(phi + k3)
        phi1 = phi + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        phi1 = jnp.clip(phi1, 0)
        phi1 = phi1 / (phi1.sum() + 1e-12)
        return phi1

    @jax.jit
    def loss_fn(params):
        A = params['A'] - jnp.diag(jnp.abs(jnp.diag(params['A'])))  # enforce diag ≤ 0
        B = params['B']  # (n_p, N_G)

        # trajectory MSE
        mse = 0.0
        for p in range(n_p):
            phi1 = replicator_step(phi_jax[p, 0], A, B[p])
            phi2 = replicator_step(phi1,            A, B[p])
            mse += jnp.mean((phi1 - phi_jax[p, 1])**2)
            mse += jnp.mean((phi2 - phi_jax[p, 2])**2)
        mse = mse / (2 * n_p)

        # sign prior penalty (soft constraint)
        sp = jnp.sign(nf_jax)
        mask = (sp != 0) & (~jnp.eye(N_G, dtype=bool))
        penalty = jnp.sum(
            jnp.where(mask,
                      jnp.abs(nf_jax) * jnp.clip(-sp * A, 0) ** 2,
                      0.0)
        ) / (2 * args.sigma ** 2)

        # L2 regularisation
        l2 = args.lam * jnp.sum(A ** 2)

        return mse + penalty + l2

    # Initialise params
    rng_np = np.random.default_rng(args.seed)
    params = {
        'A': jnp.array(rng_np.normal(0, 0.1, (N_G, N_G))),
        'B': jnp.zeros((n_p, N_G)),
    }

    opt = optax.adam(args.lr)
    opt_state = opt.init(params)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    losses = []
    for epoch in range(args.epochs):
        loss, grads = grad_fn(params)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        losses.append(float(loss))
        if (epoch + 1) % 200 == 0:
            print(f'  epoch {epoch+1:4d}  loss={loss:.6f}')

    A_final = np.array(params['A'])
    A_final -= np.diag(np.abs(np.diag(A_final)))  # enforce diagonal constraint
    B_final = np.array(params['B'])
    return A_final, B_final, losses


def run_numpy_fallback(phi_obs, net_flow, args):
    """Gradient-free fallback using scipy optimize."""
    from scipy.optimize import minimize

    n_p, n_t, _ = phi_obs.shape

    def pack(A, B): return np.concatenate([A.ravel(), B.ravel()])
    def unpack(theta):
        A = theta[:N_G*N_G].reshape(N_G, N_G)
        B = theta[N_G*N_G:].reshape(n_p, N_G)
        return A, B

    def obj(theta):
        A, B = unpack(theta)
        A = A.copy(); np.fill_diagonal(A, -np.abs(np.diag(A)))
        mse = 0.0
        for p in range(n_p):
            phi1 = replicator_euler(phi_obs[p, 0], A, B[p])
            phi2 = replicator_euler(phi1,            A, B[p])
            mse += np.mean((phi1 - phi_obs[p, 1])**2)
            mse += np.mean((phi2 - phi_obs[p, 2])**2)
        mse /= 2 * n_p
        pen = sign_penalty_np(A, net_flow, sigma=args.sigma)
        l2  = args.lam * np.sum(A**2)
        return mse + pen + l2

    theta0 = pack(np.random.default_rng(args.seed).normal(0, 0.05, (N_G, N_G)),
                  np.zeros((n_p, N_G)))
    res = minimize(obj, theta0, method='L-BFGS-B',
                   options={'maxiter': args.epochs, 'ftol': 1e-10})
    A_final, B_final = unpack(res.x)
    np.fill_diagonal(A_final, -np.abs(np.diag(A_final)))
    return A_final, B_final, [res.fun]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int,   default=2000)
    ap.add_argument('--lr',     type=float, default=3e-3)
    ap.add_argument('--sigma',  type=float, default=0.1,
                    help='Sign-prior penalty scale')
    ap.add_argument('--lam',    type=float, default=1e-3,
                    help='L2 regularisation on A')
    ap.add_argument('--seed',   type=int,   default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    phi_obs  = np.load(PHI_NPY)        # (10, 3, 10)
    net_flow = build_net_flow()

    print(f'phi_obs: {phi_obs.shape}')
    print('Building neural ODE (linear replicator + sign constraint)...')

    jax_available = False
    try:
        import jax, optax  # noqa
        jax_available = True
        print('JAX+optax available — using gradient descent.')
    except ImportError:
        print('JAX/optax not available — falling back to scipy L-BFGS-B.')

    if jax_available:
        A_neural, B_neural, losses = run_jax(phi_obs, net_flow, args)
    else:
        A_neural, B_neural, losses = run_numpy_fallback(phi_obs, net_flow, args)

    # ── Load reference gLV A for comparison ──
    with open(FIT_FILE) as f:
        fit_ref = json.load(f)
    A_glv = np.array(fit_ref['A'])

    # ── Sign agreement metrics ──
    sp = np.sign(net_flow)
    mask = (sp != 0) & (~np.eye(N_G, dtype=bool))

    def sign_agree(A, mask, sp):
        return float((np.sign(A[mask]) == sp[mask]).mean())

    sa_neural = sign_agree(A_neural, mask, sp)
    sa_glv    = sign_agree(A_glv,    mask, sp)

    # A-to-A correlation (off-diagonal)
    off = ~np.eye(N_G, dtype=bool)
    corr = np.corrcoef(A_neural[off], A_glv[off])[0, 1]

    print(f'\nSign agreement vs metabolic prior:')
    print(f'  Neural ODE : {sa_neural:.3f}')
    print(f'  gLV (ref)  : {sa_glv:.3f}')
    print(f'A_neural vs A_glv Pearson r = {corr:.3f}')

    # ── Compute RMSE on training data ──
    sq, cnt = 0, 0
    for p in range(phi_obs.shape[0]):
        phi1 = replicator_euler(phi_obs[p, 0], A_neural, B_neural[p % len(B_neural)])
        phi2 = replicator_euler(phi1,            A_neural, B_neural[p % len(B_neural)])
        sq  += np.sum((phi1 - phi_obs[p, 1])**2) + np.sum((phi2 - phi_obs[p, 2])**2)
        cnt += 2 * N_G
    rmse_neural = np.sqrt(sq / cnt)
    rmse_glv    = float(fit_ref.get('rmse', np.nan))
    print(f'RMSE — Neural: {rmse_neural:.4f}  gLV: {rmse_glv:.4f}')

    # ── Figure ──
    apply_pub_style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Panel A: Training loss
    ax = axes[0]
    if len(losses) > 1:
        ax.semilogy(losses)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Neural ODE training loss')
    else:
        ax.text(0.5, 0.5, f'scipy L-BFGS-B\nfinal loss={losses[0]:.5f}',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Optimisation (fallback)')

    # Panel B: A_neural vs A_glv scatter
    ax = axes[1]
    ax.scatter(A_glv[off], A_neural[off], s=30, alpha=0.7, color='steelblue')
    lim = max(abs(A_glv[off]).max(), abs(A_neural[off]).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'k--', lw=0.8)
    ax.set_xlabel('A_glv[i,j]')
    ax.set_ylabel('A_neural[i,j]')
    ax.set_title(f'A matrix correlation\nr={corr:.2f}')

    # Panel C: Sign agreement comparison
    ax = axes[2]
    models = ['Neural ODE\n(+ sign prior)', 'gLV fit\n(+ sign prior)']
    values = [sa_neural, sa_glv]
    colors = ['steelblue', 'tomato']
    bars = ax.bar(models, values, color=colors, alpha=0.85)
    ax.axhline(0.5, color='k', lw=0.8, ls='--', label='random')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Sign agreement with metabolic prior')
    ax.set_title('Sign agreement comparison')
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.2f}',
                ha='center', va='bottom', fontsize=9)
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig_neural_ode.pdf')
    print(f'Figure: {OUT_DIR}/fig_neural_ode.pdf')

    # ── Save results ──
    results = {
        'jax_used':       jax_available,
        'epochs':         args.epochs,
        'lr':             args.lr,
        'sigma':          args.sigma,
        'lam':            args.lam,
        'rmse_neural':    float(rmse_neural),
        'rmse_glv':       float(rmse_glv),
        'sign_agree_neural': sa_neural,
        'sign_agree_glv':    sa_glv,
        'A_corr_neural_glv': float(corr),
        'A_neural':       A_neural.tolist(),
        'guilds':         GUILD_ORDER,
        'final_loss':     float(losses[-1]),
    }
    with open(OUT_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results: {OUT_DIR}/results.json')


if __name__ == '__main__':
    main()
