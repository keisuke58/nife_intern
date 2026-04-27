#!/usr/bin/env python3
"""
Class-level generalized Lotka-Volterra replicator model for Dieckow data.

ODE (replicator form, conserves Σφ_i = 1):
  dφ_i/dt = φ_i * (f_i(φ) - <f>(φ))
  f_i(φ)  = b_i + Σ_j A_ij * φ_j
  <f>      = Σ_i φ_i * f_i

Parameters:
  A      : (N_G, N_G) shared interaction matrix
  b_all  : (n_patients, N_G) patient-specific intrinsic rates

phi_obs: (n_patients, 3, N_G)  — weeks 1, 2, 3
RMSE integrates W1→W2 and W2→W3 (Δt = 1 week each).

The Mapping:
S. oralis $\rightarrow$ Streptococcus guild
A. naeslundii $\rightarrow$ Actinomyces guild
Veillonella spp. $\rightarrow$ Veillonella guild
F. nucleatum $\rightarrow$ Fusobacteria guild (or closest equivalent)
P. gingivalis $\rightarrow$ Bacteroidia/Prevotella guild
"""

import numpy as np
from scipy.integrate import solve_ivp
import csv
from pathlib import Path

GUILD_ORDER = [
    'Actinobacteria',
    'Coriobacteriia',
    'Bacilli',
    'Clostridia',
    'Negativicutes',
    'Bacteroidia',
    'Flavobacteriia',
    'Fusobacteriia',
    'Betaproteobacteria',
    'Gammaproteobacteria',
    'Other',
]
N_G = len(GUILD_ORDER)

GUILD_COLORS = {
    'Actinobacteria': '#976832',
    'Coriobacteriia': '#b69e7c',
    'Bacilli': '#20af53',
    'Clostridia': '#98c557',
    'Negativicutes': '#703a98',
    'Bacteroidia': '#ea2323',
    'Flavobacteriia': '#f59bc1',
    'Fusobacteriia': '#fac20b',
    'Betaproteobacteria': '#25b0e1',
    'Gammaproteobacteria': '#156fb5',
    'Other': '#050608',
}

GUILD_COLORS_LIST = [GUILD_COLORS[g] for g in GUILD_ORDER]

GUILD_SHORT = {
    'Actinobacteria':    'Actin.',
    'Bacilli':           'Bacil.',
    'Bacteroidia':       'Bact.',
    'Betaproteobacteria':'β-Prot.',
    'Clostridia':        'Clost.',
    'Coriobacteriia':    'Corio.',
    'Fusobacteriia':     'Fusob.',
    'Flavobacteriia':    'Flavo.',
    'Gammaproteobacteria':'γ-Prot.',
    'Negativicutes':     'Negat.',
    'Other':             'Other',
}
GUILD_SHORT_LIST = [GUILD_SHORT[g] for g in GUILD_ORDER]

DT = 1.0   # 1 week per interval


def replicator_rhs(t, phi, b, A):
    f   = b + A @ phi
    fmean = phi @ f
    return phi * (f - fmean)


def integrate_step(phi0, b, A, dt=DT):
    """Integrate replicator ODE for one time step; clip to [0,1] and renormalise."""
    sol = solve_ivp(replicator_rhs, [0, dt], phi0, args=(b, A),
                    method='RK45', rtol=1e-6, atol=1e-8, dense_output=False)
    phi1 = np.clip(sol.y[:, -1], 0, None)
    s = phi1.sum()
    return phi1 / s if s > 1e-12 else phi0


def predict_trajectory(phi0, b, A):
    """Predict weeks 2 and 3 starting from week 1."""
    phi2 = integrate_step(phi0, b, A)
    phi3 = integrate_step(phi2, b, A)
    return phi2, phi3


def rmse_guild(A, b_all, phi_obs):
    """Mean RMSE over all patients for W2 and W3 predictions."""
    n_p = phi_obs.shape[0]
    sq_sum = 0.0
    count  = 0
    for i in range(n_p):
        phi2_pred, phi3_pred = predict_trajectory(phi_obs[i, 0], b_all[i], A)
        sq_sum += np.sum((phi2_pred - phi_obs[i, 1])**2)
        sq_sum += np.sum((phi3_pred - phi_obs[i, 2])**2)
        count  += 2 * N_G
    return np.sqrt(sq_sum / count)


# ---- pack / unpack ----
N_A = N_G * N_G


def pack(A, b_all):
    return np.concatenate([A.ravel(), b_all.ravel()])


def unpack(theta, n_p):
    A     = theta[:N_A].reshape(N_G, N_G)
    b_all = theta[N_A:].reshape(n_p, N_G)
    return A, b_all


def default_A():
    """Initialise A: small negative diagonal (self-limitation), zeros off-diagonal."""
    A = np.zeros((N_G, N_G))
    np.fill_diagonal(A, -0.1)
    return A


def _canonical_guild_name(name):
    if name is None:
        return None
    key = str(name).strip().lower()
    if not key:
        return None
    m = {g.lower(): g for g in GUILD_ORDER}
    return m.get(key)


def conet_edges_to_mask(edges_path, *, undirected=True):
    path = Path(edges_path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    with open(path, 'r', encoding='utf-8', newline='') as f:
        head = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(head, delimiters='\t,;')
        except Exception:
            dialect = csv.excel_tab
        reader = csv.DictReader(f, dialect=dialect)
        if reader.fieldnames is None:
            raise ValueError(f'No header found in {path}')

        fields = {c.strip().lower(): c for c in reader.fieldnames}
        src_col = None
        tgt_col = None
        for cand in ('source', 'src', 'node1', 'from', 'shared name', 'name'):
            if cand in fields:
                src_col = fields[cand]
                break
        for cand in ('target', 'tgt', 'node2', 'to'):
            if cand in fields:
                tgt_col = fields[cand]
                break
        if src_col is None or tgt_col is None:
            cols = reader.fieldnames
            if cols is None or len(cols) < 2:
                raise ValueError(f'Expected at least 2 columns in {path}')

            possible = [_canonical_guild_name(c) for c in cols[1:]]
            possible = [p for p in possible if p is not None]
            if len(set(possible)) >= 5:
                row_col = cols[0]
                col_guilds = []
                for c in cols[1:]:
                    g = _canonical_guild_name(c)
                    col_guilds.append(g)

                mask = np.zeros((N_G, N_G), dtype=np.int8)
                sign_prior = np.zeros((N_G, N_G), dtype=np.int8)
                np.fill_diagonal(mask, 1)

                def _set(i_, j_, s_):
                    mask[i_, j_] = 1
                    if s_ == 0:
                        return
                    cur = int(sign_prior[i_, j_])
                    if cur == 0:
                        sign_prior[i_, j_] = int(s_)
                    elif cur != int(s_):
                        sign_prior[i_, j_] = 0

                for row in reader:
                    r_guild = _canonical_guild_name(row.get(row_col))
                    if r_guild is None:
                        continue
                    i = GUILD_ORDER.index(r_guild)
                    for c, c_guild in zip(cols[1:], col_guilds):
                        if c_guild is None:
                            continue
                        try:
                            v = float(row.get(c, 0.0))
                        except Exception:
                            continue
                        if v == 0.0:
                            continue
                        j = GUILD_ORDER.index(c_guild)
                        s = 1 if v > 0 else (-1 if v < 0 else 0)
                        _set(i, j, s)
                        if undirected and i != j:
                            _set(j, i, s)

                return mask, sign_prior

            src_col, tgt_col = cols[0], cols[1]

        sign_col = None
        for cand in ('sign', 'edge sign', 'interaction', 'type'):
            if cand in fields:
                sign_col = fields[cand]
                break
        score_col = None
        for cand in ('score', 'weight', 'combined_score', 'pearson', 'spearman'):
            if cand in fields:
                score_col = fields[cand]
                break

        mask = np.zeros((N_G, N_G), dtype=np.int8)
        sign_prior = np.zeros((N_G, N_G), dtype=np.int8)
        np.fill_diagonal(mask, 1)

        def _set(i_, j_, s_):
            mask[i_, j_] = 1
            if s_ == 0:
                return
            cur = int(sign_prior[i_, j_])
            if cur == 0:
                sign_prior[i_, j_] = int(s_)
            elif cur != int(s_):
                sign_prior[i_, j_] = 0

        for row in reader:
            src = _canonical_guild_name(row.get(src_col))
            tgt = _canonical_guild_name(row.get(tgt_col))
            if src is None or tgt is None:
                continue
            i = GUILD_ORDER.index(tgt)
            j = GUILD_ORDER.index(src)

            s = 0
            if sign_col is not None:
                v = str(row.get(sign_col, '')).strip().lower()
                if v in ('+', 'positive', 'pos', 'copresence', 'co-presence', 'cooccurrence', 'co-occurrence', '1', 'true'):
                    s = 1
                elif v in ('-', 'negative', 'neg', 'mutual exclusion', 'exclusion', '-1', 'false'):
                    s = -1
            if s == 0 and score_col is not None:
                try:
                    score = float(row.get(score_col))
                    if score > 0:
                        s = 1
                    elif score < 0:
                        s = -1
                except Exception:
                    pass

            _set(i, j, s)
            if undirected and i != j:
                _set(j, i, s)

    return mask, sign_prior
