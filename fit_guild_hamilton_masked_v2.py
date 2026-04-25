#!/usr/bin/env python3
"""
fit_guild_hamilton_masked_v2.py — physically correct ψ/φ₀ initialisation.

Differences from v1 (fit_guild_hamilton_masked.py):
  - VolumeLive/TotalArea = occ_norm = Σphibar_i = Σ(φ_i·ψ_i)          [live observable]
  - occ_total = occ_norm / PerLive = Σφ_i                              [total biomass]
  - φ₀⁽⁰⁾ = 1 − occ_total  (true free space, excl. dead cells)
  - ψᵢ⁽⁰⁾ = PerLive         (CLSM live fraction → Hamilton vitality)
  - W2→W3 also uses observed occ/PerLive to rescale the initial condition

v1 used φ₀ = 1 − occ_norm (underestimates free space when PerLive < 1)
and ψ_init = 0.999 (ignores CLSM viability).

Usage (on vancouver01):
  python3 fit_guild_hamilton_masked_v2.py
  CUDA_VISIBLE_DEVICES=1 python3 fit_guild_hamilton_masked_v2.py

Output: results/dieckow_cr/fit_guild_hamilton_masked_v2.json
"""

import json, sys, time, argparse
import os
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
jax.config.update('jax_enable_x64', True)

_nife_dir = Path(__file__).resolve().parent
if not os.environ.get("TMPDIR"):
    os.environ["TMPDIR"] = str(Path.home() / "tmp")
sys.path.insert(0, str(_nife_dir))
from guild_replicator_dieckow import GUILD_ORDER
from load_structure_dieckow import load_structural_data, build_occupancy

_repo_root = Path(__file__).resolve().parents[1]
_hamilton_path = _repo_root / 'Tmcmc202601' / 'data_5species' / 'main'
sys.path.insert(0, str(_hamilton_path))
from hamilton_ode_jax_nsp import simulate_0d_nsp

# ── Guild-level interaction mask ─────────────────────────────────────────────

def build_guild_mask(n_sp, guilds):
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

    from collections import defaultdict
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
                coop[i, j] = True
                coop[j, i] = True

    np.fill_diagonal(active, True)
    return active, coop


def build_reg_weights(n_sp, guilds, lambda_active, lambda_inactive):
    active, coop = build_guild_mask(n_sp, guilds)
    lam = np.full((n_sp, n_sp), lambda_inactive)
    lam[active] = lambda_active
    np.fill_diagonal(lam, lambda_active)
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
    coop_jnp = jnp.array(coop_upper_arr, dtype=jnp.float64)

    # phi_init: absolute fractions (sum = occ_total < 1) → phi0 = 1 - occ_total
    # psi_init: PerLive scalar (CLSM viability) → ψᵢ⁽⁰⁾ = PerLive
    def eq_phi_one(A_upper, b_p, phi_init, psi_val, alpha_val):
        theta  = jnp.concatenate([A_upper, b_p])
        phibar = simulate_0d_nsp(
            theta, n_sp=n_sp, n_steps=n_steps, dt=1e-4,
            phi_init=phi_init, psi_init=psi_val,
            c_const=25.0, alpha_const=alpha_val,
        )
        eq = phibar[-1]
        s  = eq.sum()
        return jnp.where(s > 1e-10, eq / s, jnp.ones(n_sp) / n_sp)

    @jit
    def loss_fn(A_upper, b_all, phi_obs, present_mask, occ_vals, alpha_vals, per_live_vals):
        # occ_vals:      (n_patients, 3)  — VolumeLive/TotalArea (normalised, max=1)
        # per_live_vals: (n_patients, 3)  — PerLive (0-1), default 1.0
        # alpha_vals:    (n_patients, 3)  — alpha_base * (0.5 + 0.5*PerLive)
        #
        # φ_init = φ^16S × occ_norm  (absolute fracs, sum = occ_norm)
        # φ₀     = 1 - occ_norm       (free space from CLSM)
        # ψ_init = PerLive            (CLSM viability, not hardcoded 0.999)
        # W2→W3: use predicted composition × observed occ_norm[W2]
        def patient_terms(b_p, phi_p, m_p, occ_p, alpha_p, pl_p):
            phi_W1_abs  = phi_p[0] * occ_p[0]
            phi_W2_pred = eq_phi_one(A_upper, b_p, phi_W1_abs, pl_p[0], alpha_p[1])

            # W2→W3: rescale with observed W2 structural data
            phi_W2_abs  = phi_W2_pred * occ_p[1]
            phi_W3_pred = eq_phi_one(A_upper, b_p, phi_W2_abs, pl_p[1], alpha_p[2])

            m2 = m_p[1]; m3 = m_p[2]
            sq  = m2 * jnp.sum((phi_W2_pred - phi_p[1]) ** 2) \
                + m3 * jnp.sum((phi_W3_pred - phi_p[2]) ** 2)
            cnt = (m2 + m3) * n_sp
            return sq, cnt

        sq_all, cnt_all = vmap(patient_terms)(
            b_all, phi_obs, present_mask, occ_vals, alpha_vals, per_live_vals)
        sq   = jnp.sum(sq_all)
        cnt  = jnp.sum(cnt_all)
        rmse = jnp.sqrt(jnp.where(cnt > 0, sq / cnt, 0.0))
        reg      = jnp.sum(lam_jnp * A_upper ** 2)
        sign_pen = lambda_sign * jnp.sum(coop_jnp * jnp.maximum(0.0, -A_upper) ** 2)
        return rmse + reg + sign_pen

    grad_fn = jit(grad(loss_fn, argnums=(0, 1)))

    def rmse_pure(A_upper, b_all, phi_obs, present_mask, occ_vals, alpha_vals, per_live_vals):
        def patient_terms(b_p, phi_p, m_p, occ_p, alpha_p, pl_p):
            phi_W1_abs  = phi_p[0] * occ_p[0]
            phi_W2_pred = eq_phi_one(A_upper, b_p, phi_W1_abs, pl_p[0], alpha_p[1])
            phi_W2_abs  = phi_W2_pred * occ_p[1]
            phi_W3_pred = eq_phi_one(A_upper, b_p, phi_W2_abs, pl_p[1], alpha_p[2])
            m2 = m_p[1]; m3 = m_p[2]
            sq  = m2 * jnp.sum((phi_W2_pred - phi_p[1]) ** 2) \
                + m3 * jnp.sum((phi_W3_pred - phi_p[2]) ** 2)
            cnt = (m2 + m3) * n_sp
            return sq, cnt
        sq_all, cnt_all = vmap(patient_terms)(
            b_all, phi_obs, present_mask, occ_vals, alpha_vals, per_live_vals)
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
    ap.add_argument('--out-json',        default=str(Path(__file__).parent / 'results' / 'dieckow_cr' / 'fit_guild_hamilton_masked_v2.json'))
    ap.add_argument('--warm-start-json', default=str(Path(__file__).parent / 'results' / 'dieckow_cr' / 'fit_guild_hamilton_masked.json'),
                    help='Warm start from v1 masked run (or hamilton baseline)')
    ap.add_argument('--n-steps',         type=int,   default=200)
    ap.add_argument('--epochs',          type=int,   default=3000)
    ap.add_argument('--lr',              type=float, default=1e-3)
    ap.add_argument('--lambda-active',   type=float, default=1e-4)
    ap.add_argument('--lambda-inactive', type=float, default=5e-3)
    ap.add_argument('--lambda-sign',     type=float, default=1e-2)
    ap.add_argument('--log-every',       type=int,   default=100)
    ap.add_argument('--struct-xlsx',     default=str(Path(__file__).parent / 'Datasets' /
                    'Abutment_Structure vs composition.xlsx'))
    ap.add_argument('--alpha-base',      type=float, default=100.0)
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

    # ── Structural data ───────────────────────────────────────────────────────
    struct_path = Path(args.struct_xlsx)
    if struct_path.exists():
        struct_data  = load_structural_data(struct_path)
        occ, max_occ = build_occupancy(struct_data, normalize=True)
        per_live_raw = struct_data.get('PerLive', {})
        n_p_keep, n_w = phi_obs.shape[0], 3
        occ_arr      = np.ones((n_p_keep, n_w))
        per_live_arr = np.ones((n_p_keep, n_w))   # default: all live (PerLive=1)
        alpha_arr    = np.full((n_p_keep, n_w), args.alpha_base)
        for p_idx, pat in enumerate(patients):
            for w in range(n_w):
                key = (pat, w + 1)
                occ_arr[p_idx, w]      = occ.get(key, 1.0)
                pl_val = per_live_raw.get(key, 100.0) / 100.0
                per_live_arr[p_idx, w] = pl_val
                alpha_arr[p_idx, w]    = args.alpha_base * (0.5 + 0.5 * pl_val)
        print(f'Structural data loaded (max_occ={max_occ:.4f} µm³/µm²)', flush=True)
        print(f'  occ_norm  range: [{occ_arr.min():.3f}, {occ_arr.max():.3f}]', flush=True)
        print(f'  per_live  range: [{per_live_arr.min():.3f}, {per_live_arr.max():.3f}]', flush=True)
        print(f'  phi0_init range: [{(1-occ_arr).min():.3f}, {(1-occ_arr).max():.3f}]', flush=True)
        print(f'  psi_init  range (=PerLive): [{per_live_arr.min():.3f}, {per_live_arr.max():.3f}]', flush=True)
        print(f'  alpha     range: [{alpha_arr.min():.1f}, {alpha_arr.max():.1f}]', flush=True)
    else:
        print('[WARN] struct-xlsx not found, using occ=1, PerLive=1, fixed alpha', flush=True)
        occ_arr      = np.ones((phi_obs.shape[0], 3))
        per_live_arr = np.ones((phi_obs.shape[0], 3))
        alpha_arr    = np.full((phi_obs.shape[0], 3), args.alpha_base)

    occ_vals      = jnp.array(occ_arr)
    per_live_vals = jnp.array(per_live_arr)
    alpha_vals    = jnp.array(alpha_arr)

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

    # Warm start — try v1 masked, then baseline hamilton
    warm_paths = [Path(args.warm_start_json),
                  Path(args.warm_start_json).parent / 'fit_guild_hamilton.json']
    warm = next((p for p in warm_paths if p.exists()), None)
    if warm is not None:
        d = json.load(open(warm))
        A0 = np.array(d['A'])
        if A0.shape[0] < n_sp:
            A_pad = np.zeros((n_sp, n_sp))
            A_pad[:A0.shape[0], :A0.shape[0]] = A0
            np.fill_diagonal(A_pad[A0.shape[0]:, A0.shape[0]:], -0.1)
            A0 = A_pad
        elif A0.shape[0] > n_sp:
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
        print(f'Warm start: loaded {warm.name}', flush=True)
    else:
        A_upper = default_A_upper(n_sp)
        b_all   = jnp.full((phi_obs.shape[0], n_sp), 0.1)
        print('Cold start', flush=True)

    m = (jnp.zeros_like(A_upper), jnp.zeros_like(b_all))
    v = (jnp.zeros_like(A_upper), jnp.zeros_like(b_all))

    print('\nJIT compile (forward)...', flush=True)
    t0 = time.time()
    _ = loss_fn(A_upper, b_all, phi_obs, present_mask, occ_vals, alpha_vals, per_live_vals)
    print(f'Forward done in {time.time()-t0:.1f}s', flush=True)
    print('Compiling grad...', flush=True)
    _ = grad_fn(A_upper, b_all, phi_obs, present_mask, occ_vals, alpha_vals, per_live_vals)
    print(f'Grad compiled in {time.time()-t0:.1f}s', flush=True)

    best_loss = float(loss_fn(A_upper, b_all, phi_obs, present_mask, occ_vals, alpha_vals, per_live_vals))
    best_A, best_b = A_upper, b_all

    for epoch in range(1, args.epochs + 1):
        gA, gb = grad_fn(A_upper, b_all, phi_obs, present_mask, occ_vals, alpha_vals, per_live_vals)
        (A_upper, b_all), m, v = adam_step(
            (A_upper, b_all), (gA, gb), m, v, epoch, lr=args.lr)
        A_upper = apply_diag_constraint(A_upper, n_sp)

        if epoch % args.log_every == 0 or epoch == 1:
            val = float(loss_fn(A_upper, b_all, phi_obs, present_mask, occ_vals, alpha_vals, per_live_vals))
            print(f'  epoch {epoch:5d}  loss={val:.5f}  ({time.time()-t0:.1f}s)', flush=True)
            if val < best_loss:
                best_loss = val
                best_A, best_b = A_upper, b_all

    A_upper, b_all = best_A, best_b
    rmse = rmse_pure(A_upper, b_all, phi_obs, present_mask, occ_vals, alpha_vals, per_live_vals)
    print(f'\nFinal RMSE: {rmse:.5f}  ({time.time()-t0:.1f}s)', flush=True)

    A_full = np.zeros((n_sp, n_sp))
    idx = 0
    for j in range(n_sp):
        for i in range(j + 1):
            v_ = float(A_upper[idx])
            A_full[i, j] = v_
            A_full[j, i] = v_
            idx += 1

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
        occ_vals=occ_arr.tolist(),
        per_live_vals=per_live_arr.tolist(),
        alpha_vals=alpha_arr.tolist(),
        alpha_base=args.alpha_base,
        message=(
            'Hamilton masked+struct v2 (phi0=1-occ_total, psi_init=PerLive, '
            f'W2->W3 uses obs occ) Adam lr={args.lr} '
            f'lam_act={args.lambda_active} lam_inact={args.lambda_inactive}'
        ),
    ), open(out, 'w'), indent=2)
    print(f'Saved: {out}', flush=True)


if __name__ == '__main__':
    main()
