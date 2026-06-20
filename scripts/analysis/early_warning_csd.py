"""Early-warning signals (critical slowing down) for the peri-implant / periodontal dysbiosis tipping.

Premise. The mechano-biological vicious cycle has a tipping point (a fold/saddle-node: the healthy
commensal attractor collapses as dysbiosis pressure rises). Dynamical-systems theory predicts that as
a system approaches such a fold, it recovers more and more slowly from perturbations — "critical
slowing down" — so the *fluctuations* of the state show rising lag-1 autocorrelation (AR1) and
variance BEFORE the transition. If real microbiome time series carry this signature, the tipping point
stops being a model artefact and becomes a *forecastable* clinical phenomenon.

This script:
  (A) MODEL — a minimal stochastic model of the dysbiosis fold (May grazing form) with a slowly rising
      dysbiosis pressure; shows AR1 and variance rising and peaking before the fold. Establishes the
      CSD signature in our class of system.
  (B) DATA — the Duran-Pinedo 2021 longitudinal cohort (15 patients x 7 timepoints, GDI(t)). Per
      patient we detrend GDI(t) and compute its lag-1 AR1 and variance, then test across the cohort
      whether patients heading TOWARD dysbiosis (larger GDI increase) carry stronger CSD indicators.

HONEST SCOPE. 7 timepoints is short for robust CSD (the method wants longer series); the data part is
exploratory proof-of-concept, not a definitive test. The cohort is periodontitis subgingival plaque
(same dysbiosis dynamics, not peri-implant) and aggregated across site types.

Run:  python scripts/analysis/early_warning_csd.py
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import kendalltau, spearmanr

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "prjna725874"
OUT = ROOT / "masterarbeit_ansys_fem" / "figures"
OUT.mkdir(exist_ok=True)
DYS, COM = [2, 4, 6], [0, 1, 8]          # dysbiotic / commensal guild indices (GUILD_ORDER)
RNG = np.random.default_rng(0)


def ar1(x):
    """Lag-1 autocorrelation of a 1-D series."""
    x = np.asarray(x, float) - np.mean(x)
    if len(x) < 3 or np.allclose(x, 0):
        return np.nan
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def rolling_ews(series, win):
    """Rolling lag-1 AR1 and variance on a Gaussian-detrended series."""
    s = np.asarray(series, float)
    resid = s - gaussian_filter1d(s, sigma=max(1.0, len(s) / 12))
    ar, var, idx = [], [], []
    for i in range(win, len(s) + 1):
        w = resid[i - win:i]
        ar.append(ar1(w)); var.append(np.var(w)); idx.append(i - 1)
    return np.array(idx), np.array(ar), np.array(var)


# ── (A) model: stochastic dysbiosis fold (May grazing form), rising pressure c(t) ───────────────────
def _one_run(K, h, T, dt, sigma, c, rng):
    x = np.empty(T); x[0] = 8.0
    for k in range(1, T):
        drift = x[k - 1] * (1 - x[k - 1] / K) - c[k] * x[k - 1] ** 2 / (x[k - 1] ** 2 + h ** 2)
        x[k] = max(1e-3, x[k - 1] + drift * dt + sigma * np.sqrt(dt) * rng.standard_normal())
    return x


def model_csd(n_ens=60):
    """Ensemble-averaged CSD: AR1 & variance computed in a rolling window on each realisation, then
    averaged over the ensemble vs the (pre-tip) dysbiosis pressure — the standard smooth EWS picture."""
    K, h = 10.0, 1.0
    T, dt, sigma = 4000, 0.02, 0.16
    c = np.linspace(1.6, 2.62, T)          # ramp up to just before the fold (~c*≈2.6); avoid post-collapse
    cg = np.linspace(c[0], c[-1], 40)      # common pressure grid for ensemble averaging
    AR, VR = [], []
    x_demo = None
    for j in range(n_ens):
        x = _one_run(K, h, T, dt, sigma, c, np.random.default_rng(j))
        if x_demo is None:
            x_demo = x
        # rolling EWS over the full ramp, mapped onto the c grid
        idx, a, v = rolling_ews(x[::T // 240], win=24)
        cc = np.interp(idx, np.arange(len(x[::T // 240])), np.linspace(c[0], c[-1], len(x[::T // 240])))
        AR.append(np.interp(cg, cc, a)); VR.append(np.interp(cg, cc, v))
    arr = np.nanmean(AR, 0); var = np.nanmean(VR, 0)
    return c, x_demo, cg, arr, var


# ── (B) data: Duran-Pinedo GDI(t) per patient ───────────────────────────────────────────────────────
def data_csd():
    phi = np.load(DATA / "phi_guild.npy")                       # (P, T, G)
    gdi = (np.log(phi[:, :, DYS].sum(2) + 1e-6) - np.log(phi[:, :, COM].sum(2) + 1e-6))
    P, T = gdi.shape
    rows = []
    for p in range(P):
        g = gdi[p]
        resid = g - gaussian_filter1d(g, sigma=1.0)
        rows.append({
            "ar1": ar1(resid),
            "var": float(np.var(resid)),
            "gdi_change": float(g[-1] - g[0]),                  # toward dysbiosis if > 0
            "gdi_slope": float(np.polyfit(np.arange(T), g, 1)[0]),
        })
    return rows


def main():
    c_full, x_demo, cg, arr, var = model_csd()
    rows = data_csd()
    ar = np.array([r["ar1"] for r in rows]); vr = np.array([r["var"] for r in rows])
    dg = np.array([r["gdi_change"] for r in rows])
    ok = np.isfinite(ar)
    rho_ar, p_ar = spearmanr(dg[ok], ar[ok])
    rho_vr, p_vr = spearmanr(dg, vr)
    print(f"(A) model (ensemble): AR1 {arr[0]:.2f}->{np.nanmax(arr):.2f}, variance x{np.nanmax(var)/var[0]:.1f} toward the fold")
    print(f"(B) data (n={ok.sum()}): Spearman(GDI change, AR1)  = {rho_ar:+.2f}  p={p_ar:.3f}")
    print(f"    Spearman(GDI change, variance) = {rho_vr:+.2f}  p={p_vr:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    # (A) model state + AR1
    ax = axes[0]; ax.plot(c_full, x_demo, color="#2c3e50", lw=0.8, alpha=0.8, label="community state")
    ax.set_xlabel("dysbiosis pressure $c$ (ramping)"); ax.set_ylabel("commensal state $x$")
    ax2 = ax.twinx()
    ax2.plot(cg, arr, color="#c0392b", lw=2.2, label="AR1 (ensemble)"); ax2.set_ylabel("lag-1 AR1", color="#c0392b")
    ax.set_title("(A) model: state declines, AR1 rises", fontsize=11, loc="left")
    ax.legend(fontsize=8, loc="lower left"); ax.grid(True, lw=0.4, alpha=0.4)
    # (B) model EWS rising (ensemble-averaged, smooth)
    ax = axes[1]
    ax.plot(cg, (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr) + 1e-9), color="#c0392b", lw=2.2, label="AR1 (norm)")
    ax.plot(cg, (var - np.nanmin(var)) / (np.nanmax(var) - np.nanmin(var) + 1e-9), color="#e67e22", lw=2.2, label="variance (norm)")
    ax.set_xlabel("dysbiosis pressure $c$ (→ fold)"); ax.set_ylabel("early-warning indicator (norm.)")
    ax.set_title("(B) CSD: AR1 & variance rise before the tip", fontsize=11, loc="left")
    ax.legend(fontsize=8); ax.grid(True, lw=0.4, alpha=0.4)
    # (B) data: AR1 vs GDI change
    ax = axes[2]
    sc = ax.scatter(dg[ok], ar[ok], c=vr[ok], cmap="viridis", s=55, edgecolor="k", lw=0.4)
    ax.axvline(0, color="0.6", ls="--", lw=0.8)
    ax.set_xlabel("GDI change over 12 mo  (→ dysbiosis if > 0)"); ax.set_ylabel("lag-1 AR1 of GDI(t)")
    ax.set_title(f"(C) data: AR1 vs trajectory\nSpearman $\\rho$={rho_ar:+.2f}, p={p_ar:.2f} (n={ok.sum()})",
                 fontsize=10, loc="left")
    fig.colorbar(sc, ax=ax, label="variance of GDI(t)")
    ax.grid(True, lw=0.4, alpha=0.4)

    fig.suptitle("Critical slowing down as an early-warning signal of the dysbiosis tipping point",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_early_warning_csd.{ext}", dpi=200)
    json.dump({"model_ar1_range": [float(arr[0]), float(np.nanmax(arr))],
               "data_spearman_ar1": [rho_ar, p_ar], "data_spearman_var": [rho_vr, p_vr],
               "n": int(ok.sum())}, open(ROOT / "results" / "early_warning_csd.json", "w"), indent=2)
    print(f"\nwrote {(OUT / 'fig_early_warning_csd.pdf').name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
