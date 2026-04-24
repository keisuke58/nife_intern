#!/usr/bin/env python3
"""
Guild-level Hamilton ODE fit with structured regularization.

Improvements over fit_guild_hamilton_gpu.py:
  1. Parameter masking: non-active A links get stronger L2 (lambda_inactive)
  2. Sign prior: known cooperative links penalised if negative (lambda_sign)
  3. Interaction mask built from Dieckow SI Relationships xlsx

Active links are guild pairs with evidence of metabolite cross-feeding in the
Dieckow SI xlsx (PRODUCES/USES relationships aggregated to class level).

Usage (on vancouver01):
  CUDA_VISIBLE_DEVICES=1 python3 fit_guild_hamilton_masked.py

Output: results/dieckow_cr/fit_guild_hamilton_masked.json
"""

import json, sys, time, argparse
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
jax.config.update('jax_enable_x64', True)

_nife_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_nife_dir))
from guild_replicator_dieckow import GUILD_ORDER

_repo_root = Path(__file__).resolve().parents[1]
_hamilton_path = _repo_root / 'Tmcmc202601' / 'data_5species' / 'main'
sys.path.insert(0, str(_hamilton_path))
from hamilton_ode_jax_nsp import simulate_0d_nsp

# ── Guild-level interaction mask ─────────────────────────────────────────────

def build_guild_mask(n_sp, guilds):
    """Build (n_sp, n_sp) mask of biologically active links from SI xlsx.

    Returns:
        active_sym: bool (n_sp, n_sp) — symmetric mask of known cross-feeding pairs
        coop_sym:   bool (n_sp, n_sp) — cooperative (expected +) subset of active
    """
    try:
        import openpyxl
        SI_XLSX = Path(__file__).parent / 'Datasets' / '20260416_AbutmentPapernpjBiofilmsDieckow_SI_Relationships.xlsx'
        active_sym, coop_sym = _load_mask_from_xlsx(SI_XLSX, guilds)
        return active_sym, coop_sym
    except Exception as e:
        print(f'[WARN] Could not load SI xlsx ({e}), using uniform mask.', flush=True)
        active_sym = np.ones((n_sp, n_sp), dtype=bool)
        coop_sym   = np.zeros((n_sp, n_sp), dtype=bool)
        return active_sym, coop_sym


def _load_mask_from_xlsx(xlsx_path, guilds):
    import openpyxl
    from collections import defaultdict

    GUILD_MAP = {
        'Abiotrophia': 'Bacilli', 'Aerococcus': 'Bacilli', 'Gemella': 'Bacilli',
        'Granulicatella': 'Bacilli', 'Lacticaseibacillus': 'Bacilli',
        'Lactiplantibacillus': 'Bacilli', 'Limosilactobacillus': 'Bacilli',
        'Streptococcus': 'Bacilli',
        'Actinomyces': 'Actinobacteria', 'Bifidobacterium': 'Actinobacteria',
        'Corynebacterium': 'Actinobacteria', 'Rothia': 'Actinobacteria',
        'Slackia': 'Actinobacteria',
        'Alloprevotella': 'Bacteroidia', 'Porphyromonas': 'Bacteroidia',
        'Prevotella': 'Bacteroidia', 'Prevotella_7': 'Bacteroidia',
        'Tannerella': 'Bacteroidia',
        'Aggregatibacter': 'Betaproteobacteria', 'Cardiobacterium': 'Betaproteobacteria',
        'Eikenella': 'Betaproteobacteria', 'Kingella': 'Betaproteobacteria',
        'Neisseria': 'Betaproteobacteria',
        'Anaerococcus': 'Clostridia', 'Catonella': 'Clostridia',
        'Finegoldia': 'Clostridia', 'Johnsonella': 'Clostridia',
        'Lachnoanaerobaculum': 'Clostridia', 'Mogibacterium': 'Clostridia',
        'Oribacterium': 'Clostridia', 'Parvimonas': 'Clostridia',
        'Peptoniphilus': 'Clostridia', 'Peptostreptococcus': 'Clostridia',
        'Solobacterium': 'Clostridia', 'Stomatobaculum': 'Clostridia',
        'Atopobium': 'Coriobacteriia', 'Cryptobacterium': 'Coriobacteriia',
        'Olsenella': 'Coriobacteriia',
        'Fusobacterium': 'Fusobacteriia', 'Leptotrichia': 'Fusobacteriia',
        'Capnocytophaga': 'Flavobacteriia', 'Bergeyella': 'Flavobacteriia',
        'Riemerella': 'Flavobacteriia',
        'Haemophilus': 'Gammaproteobacteria', 'Pseudomonas': 'Gammaproteobacteria',
        'Centipeda': 'Negativicutes', 'Dialister': 'Negativicutes',
        'Megasphaera': 'Negativicutes', 'Selenomonas': 'Negativicutes',
        'Veillonella': 'Negativicutes',
        'Campylobacter': 'Other', 'Treponema': 'Other',
        'Shuttleworthia': 'Other',
    }

    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    header = rows[0]
    hidx = {h: i for i, h in enumerate(header)}

    # met → guild → [PRODUCES|USES]
    met_guild = defaultdict(lambda: defaultdict(list))
    for r in rows[1:]:
        if not r[hidx['TAXON']]: continue
        genus = str(r[hidx['TAXON']]).strip().split()[0]
        guild = GUILD_MAP.get(genus)
        if not guild or guild not in guilds: continue
        rel = str(r[hidx['RELATIONSHIP']]).strip().upper()
        obj = str(r[hidx['OBJECT']]).strip().lower()
        otype = str(r[hidx['OBJECT_TYPE']]).strip() if r[hidx['OBJECT_TYPE']] else 'metabolite'
        if rel in ('PRODUCES', 'USES') and otype in ('metabolite', 'enzyme'):
            met_guild[obj][guild].append(rel)

    gi = {g: i for i, g in enumerate(guilds)}
    n = len(guilds)
    active = np.zeros((n, n), dtype=bool)
    coop   = np.zeros((n, n), dtype=bool)

    for met, gd in met_guild.items():
        producers = [g for g, rels in gd.items() if 'PRODUCES' in rels]
        consumers = [g for g, rels in gd.items() if 'USES' in rels]
        for p in producers:
            for c in consumers:
                if p == c: continue
                i, j = gi[p], gi[c]
                active[i, j] = True
                active[j, i] = True
                coop[i, j] = True  # cross-feeding → expect positive A
                coop[j, i] = True

    # Always keep diagonal active (self-inhibition always present)
    np.fill_diagonal(active, True)
    return active, coop


def build_reg_weights(n_sp, guilds, lambda_active, lambda_inactive):
    """Return (n_upper,) array of per-entry lambda_reg values.

    Active links → lambda_active; inactive → lambda_inactive.
    Diagonal always gets lambda_active (but constrained ≤ 0 separately).
    """
    active, coop = build_guild_mask(n_sp, guilds)
    lam = np.full((n_sp, n_sp), lambda_inactive)
    lam[active] = lambda_active
    np.fill_diagonal(lam, lambda_active)

    # Upper triangle (column-major, including diagonal)
    lam_upper = []
    coop_upper = []
    for j in range(n_sp):
        for i in range(j + 1):
            lam_upper.append(lam[i, j])
            coop_upper.append(bool(coop[i, j] and i != j))
    return np.array(lam_upper), np.array(coop_upper, dtype=bool)


# ── Adam + loss ───────────────────────────────────────────────────────────────

def build_fns(n_sp, n_steps, lam_upper_arr, coop_upper_arr, lambda_sign):
    lam_jnp  = jnp.array(lam_upper_arr)
    # sign penalty: for cooperative links, penalise if A < 0
    # for all other off-diagonal links we skip sign penalty
    coop_jnp = jnp.array(coop_upper_arr, dtype=jnp.float64)

    @jit
    def eq_phi_one(A_upper, b_p, phi_init):
        theta  = jnp.concatenate([A_upper, b_p])
        s0 = jnp.sum(phi_init)
        phi0 = jnp.where(s0 > 1e-12, phi_init / s0, jnp.ones(n_sp) / n_sp)
        phibar = simulate_0d_nsp(
            theta, n_sp=n_sp, n_steps=n_steps, dt=1e-4, phi_init=phi0,
            c_const=25.0, alpha_const=100.0
        )
        eq = phibar[-1]
        s  = eq.sum()
        return jnp.where(s > 1e-10, eq / s, jnp.ones(n_sp) / n_sp)

    @jit
    def loss_fn(A_upper, b_all, phi_obs, present_mask):
        def patient_terms(b_p, phi_p, m_p):
            phi_W2 = eq_phi_one(A_upper, b_p, phi_p[0])
            phi_W3 = eq_phi_one(A_upper, b_p, phi_W2)
            m2 = m_p[1]; m3 = m_p[2]
            sq = m2 * jnp.sum((phi_W2 - phi_p[1]) ** 2) + m3 * jnp.sum((phi_W3 - phi_p[2]) ** 2)
            cnt = (m2 + m3) * n_sp
            return sq, cnt

        sq_all, cnt_all = vmap(patient_terms)(b_all, phi_obs, present_mask)
        sq   = jnp.sum(sq_all)
        cnt  = jnp.sum(cnt_all)
        rmse = jnp.sqrt(jnp.where(cnt > 0, sq / cnt, 0.0))
        # Structured L2: per-link lambda
        reg = jnp.sum(lam_jnp * A_upper ** 2)
        # Sign prior: penalise cooperative links that go negative
        sign_pen = lambda_sign * jnp.sum(coop_jnp * jnp.maximum(0.0, -A_upper) ** 2)
        return rmse + reg + sign_pen

    grad_fn = jit(grad(loss_fn, argnums=(0, 1)))

    def rmse_pure(A_upper, b_all, phi_obs, present_mask):
        def patient_terms(b_p, phi_p, m_p):
            phi_W2 = eq_phi_one(A_upper, b_p, phi_p[0])
            phi_W3 = eq_phi_one(A_upper, b_p, phi_W2)
            m2 = m_p[1]; m3 = m_p[2]
            sq = m2 * jnp.sum((phi_W2 - phi_p[1]) ** 2) + m3 * jnp.sum((phi_W3 - phi_p[2]) ** 2)
            cnt = (m2 + m3) * n_sp
            return sq, cnt
        sq_all, cnt_all = vmap(patient_terms)(b_all, phi_obs, present_mask)
        sq = float(jnp.sum(sq_all)); cnt = float(jnp.sum(cnt_all))
        return float(np.sqrt(sq / cnt)) if cnt > 0 else float('nan')

    return eq_phi_one, loss_fn, grad_fn, rmse_pure


def adam_step(params, grads, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    m_new = tuple(b1 * mi + (1 - b1) * gi for mi, gi in zip(m, grads))
    v_new = tuple(b2 * vi + (1 - b2) * gi**2 for vi, gi in zip(v, grads))
    mh = tuple(mi / (1 - b1**t) for mi in m_new)
    vh = tuple(vi / (1 - b2**t) for vi in v_new)
    p_new = tuple(pi - lr * mhi / (jnp.sqrt(vhi) + eps)
                  for pi, mhi, vhi in zip(params, mh, vh))
    return p_new, m_new, v_new


def apply_diag_constraint(A_upper, n_sp):
    """Diagonal entries ≤ 0."""
    idx = 0
    for j in range(n_sp):
        for i in range(j + 1):
            if i == j:
                A_upper = A_upper.at[idx].set(jnp.minimum(A_upper[idx], 0.0))
            idx += 1
    return A_upper


def default_A_upper(n_sp):
    A = -0.1 * np.eye(n_sp)
    upper = []
    for j in range(n_sp):
        for i in range(j + 1):
            upper.append(A[i, j])
    return jnp.array(upper)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phi-npy',         default=str(Path(__file__).parent / 'results' / 'dieckow_otu' / 'phi_guild.npy'))
    ap.add_argument('--out-json',        default=str(Path(__file__).parent / 'results' / 'dieckow_cr' / 'fit_guild_hamilton_masked.json'))
    ap.add_argument('--warm-start-json', default=str(Path(__file__).parent / 'results' / 'dieckow_cr' / 'fit_guild_hamilton.json'))
    ap.add_argument('--n-steps',         type=int,   default=200)
    ap.add_argument('--epochs',          type=int,   default=3000)
    ap.add_argument('--lr',              type=float, default=1e-3)
    ap.add_argument('--lambda-active',   type=float, default=1e-4,
                    help='L2 penalty for biologically active A links')
    ap.add_argument('--lambda-inactive', type=float, default=5e-3,
                    help='L2 penalty for inactive (no SI evidence) links')
    ap.add_argument('--lambda-sign',     type=float, default=1e-2,
                    help='Sign penalty for cooperative links that go negative')
    ap.add_argument('--log-every',       type=int,   default=100)
    args = ap.parse_args()

    print(f'JAX devices: {jax.devices()}', flush=True)

    phi_all = np.load(args.phi_npy)
    if phi_all.ndim != 3 or phi_all.shape[1] != 3:
        raise ValueError(f'Expected (n_patients,3,n_guilds), got {phi_all.shape}')
    n_p, _, n_sp = phi_all.shape
    guilds = GUILD_ORDER[:n_sp]

    present = (phi_all.sum(axis=2) > 1e-12).astype(np.float64)
    keep    = present[:, 0] > 0.0
    phi_all = phi_all[keep]
    present = present[keep]
    patients = ([p for k, p in zip(keep.tolist(), list('ABCDEFGHKL')) if k]
                if n_p == 10 else [str(i) for i in range(phi_all.shape[0])])

    phi_obs      = jnp.array(phi_all)
    present_mask = jnp.array(present)
    print(f'phi_obs: {phi_obs.shape}', flush=True)

    # Build structured reg weights from SI xlsx
    lam_upper, coop_upper = build_reg_weights(n_sp, guilds, args.lambda_active, args.lambda_inactive)
    n_active   = np.sum(lam_upper < (args.lambda_active + args.lambda_inactive) / 2)
    n_inactive = np.sum(lam_upper >= (args.lambda_active + args.lambda_inactive) / 2)
    n_coop     = np.sum(coop_upper)
    print(f'Links — active: {n_active}, inactive: {n_inactive}, cooperative: {n_coop}', flush=True)

    eq_phi_one, loss_fn, grad_fn, rmse_pure = build_fns(
        n_sp=n_sp, n_steps=args.n_steps,
        lam_upper_arr=lam_upper, coop_upper_arr=coop_upper,
        lambda_sign=args.lambda_sign,
    )

    # Warm start from previous Hamilton fit
    warm = Path(args.warm_start_json)
    if warm.exists():
        d = json.load(open(warm))
        A0 = np.array(d['A'])
        if A0.shape[0] != n_sp:
            A0 = A0[:n_sp, :n_sp]
        A_sym = (A0 + A0.T) / 2.0
        A_upper_init = []
        for j in range(n_sp):
            for i in range(j + 1):
                A_upper_init.append(A_sym[i, j])
        A_upper = jnp.array(A_upper_init)
        b0 = np.array(d['b_all'])
        n_keep = int(keep.sum())
        if b0.shape[0] == n_keep:
            b_all = jnp.array(b0[:, :n_sp])
        elif b0.shape[0] >= n_p:
            b_all = jnp.array(b0[:n_p, :n_sp][keep, :])
        else:
            b_all = jnp.full((n_keep, n_sp), 0.1)
        print('Warm start: loaded fit_guild_hamilton.json', flush=True)
    else:
        A_upper = default_A_upper(n_sp)
        b_all   = jnp.full((phi_obs.shape[0], n_sp), 0.1)
        print('Cold start', flush=True)

    m = (jnp.zeros_like(A_upper), jnp.zeros_like(b_all))
    v = (jnp.zeros_like(A_upper), jnp.zeros_like(b_all))

    print('\nJIT compile (forward)...', flush=True)
    t0 = time.time()
    _ = loss_fn(A_upper, b_all, phi_obs, present_mask)
    print(f'Forward done in {time.time()-t0:.1f}s', flush=True)
    print('Compiling grad...', flush=True)
    _ = grad_fn(A_upper, b_all, phi_obs, present_mask)
    print(f'Grad compiled in {time.time()-t0:.1f}s', flush=True)

    best_loss = float(loss_fn(A_upper, b_all, phi_obs, present_mask))
    best_A, best_b = A_upper, b_all

    for epoch in range(1, args.epochs + 1):
        gA, gb = grad_fn(A_upper, b_all, phi_obs, present_mask)
        (A_upper, b_all), m, v = adam_step(
            (A_upper, b_all), (gA, gb), m, v, epoch, lr=args.lr)
        A_upper = apply_diag_constraint(A_upper, n_sp)

        if epoch % args.log_every == 0 or epoch == 1:
            val = float(loss_fn(A_upper, b_all, phi_obs, present_mask))
            print(f'  epoch {epoch:5d}  loss={val:.5f}  ({time.time()-t0:.1f}s)', flush=True)
            if val < best_loss:
                best_loss = val
                best_A, best_b = A_upper, b_all

    A_upper, b_all = best_A, best_b
    rmse = rmse_pure(A_upper, b_all, phi_obs, present_mask)
    print(f'\nFinal RMSE: {rmse:.5f}  ({time.time()-t0:.1f}s)', flush=True)

    # Reconstruct full A matrix
    A_full = np.zeros((n_sp, n_sp))
    idx = 0
    for j in range(n_sp):
        for i in range(j + 1):
            v_ = float(A_upper[idx])
            A_full[i, j] = v_
            A_full[j, i] = v_
            idx += 1

    # Report sign accuracy on cooperative links
    coop_mask_full = np.zeros((n_sp, n_sp), dtype=bool)
    _, coop_sym = build_guild_mask(n_sp, guilds)
    n_coop_pos = int(np.sum(A_full[coop_sym] > 0))
    n_coop_tot = int(np.sum(coop_sym))
    print(f'Cooperative links positive: {n_coop_pos}/{n_coop_tot}', flush=True)

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(
        A=A_full.tolist(),
        A_upper=np.array(A_upper).tolist(),
        b_all=np.array(b_all).tolist(),
        rmse=rmse,
        guilds=guilds,
        patients=patients,
        n_steps=args.n_steps,
        lambda_active=args.lambda_active,
        lambda_inactive=args.lambda_inactive,
        lambda_sign=args.lambda_sign,
        coop_positive=f'{n_coop_pos}/{n_coop_tot}',
        message=f'Hamilton masked Adam lr={args.lr} lam_act={args.lambda_active} lam_inact={args.lambda_inactive}',
    ), open(out, 'w'), indent=2)
    print(f'Saved: {out}', flush=True)


if __name__ == '__main__':
    main()
