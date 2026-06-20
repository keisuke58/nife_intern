"""PICF inverse problem — infer a patient's risk from peri-implant crevicular-fluid RANKL:OPG.

The UQ sensitivity (fig_periimplantitis_uq.py) showed the osteoclast / RANKL:OPG axis drives the
36-month-loss uncertainty, and that the highest-value datum is a PICF RANKL:OPG measurement. This
script turns that value-of-information statement into an actual estimator: given a few noisy PICF
RANKL:OPG readings, run Bayesian inference over the latent dysbiosis drive B and propagate to a
posterior over 36-month bone loss. More measurements -> a tighter risk estimate.

Forward model: the calibrated cascade (fig_periimplantitis_rankl_opg.py) maps B to (i) the observable
RANKL:OPG ratio at the clinic visit and (ii) the 36-month bone loss. Inference: grid Bayes over B with
a Gaussian measurement likelihood; posterior propagated through the outcome map.

Run: python masterarbeit_ansys_fem/extensions/picf_inverse.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "figures"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))
from fig_periimplantitis_rankl_opg import amplification, integrate, load_patients  # noqa: E402

MEAS_WEEK = 12          # PICF sampled at the ~3-month review
RNG = np.random.default_rng(1)


def forward(B, A):
    """Return (observable RANKL:OPG at the clinic visit, 36-mo bone loss) for dysbiosis drive B."""
    t, out = integrate(B, A)
    k = int(np.clip(MEAS_WEEK / (t[-1] / (len(t) - 1)), 0, len(t) - 1))
    ratio = out[k, 1] / max(out[k, 2], 1e-6)          # RANKL / OPG
    return ratio, out[-1, 5]


def main() -> int:
    A = amplification()
    _, Bdata, _ = load_patients()
    Bgrid = np.linspace(0.0, float(Bdata.max()) * 1.1, 80)
    obs_map = np.array([forward(b, A)[0] for b in Bgrid])     # RANKL:OPG(B)
    loss_map = np.array([forward(b, A)[1] for b in Bgrid])    # 36-mo loss(B)

    # synthetic "true" patient (a moderate-high progressor) + noisy PICF measurements
    B_true = float(Bdata.max()) * 0.8
    r_true, loss_true = forward(B_true, A)
    sigma = 0.12 * r_true                                     # ~12% assay noise
    meas = r_true + sigma * RNG.standard_normal(5)

    def posterior(n_meas):
        o = meas[:n_meas]
        # Gaussian likelihood of the n measurements given B (via obs_map), uniform prior on B
        ll = -0.5 * ((obs_map[None, :] - o[:, None]) / sigma) ** 2
        post = np.exp(ll.sum(0) - ll.sum(0).max())
        post /= post.sum() * (Bgrid[1] - Bgrid[0])        # normalise to a density
        return post

    fig, axes = plt.subplots(1, 4, figsize=(18.5, 4.3))

    # (A) forward maps: the observable and the outcome vs B
    ax = axes[0]; ax.plot(Bgrid, obs_map, color="#2c3e50", lw=2)
    ax.set_xlabel("latent dysbiosis drive $B$"); ax.set_ylabel("PICF RANKL:OPG (observable)", color="#2c3e50")
    ax2 = ax.twinx(); ax2.plot(Bgrid, loss_map, color="#c0392b", lw=2, ls="--")
    ax2.set_ylabel("36-mo bone loss [mm]", color="#c0392b")
    ax.axvline(B_true, color="0.6", ls=":", lw=1.0)
    ax.set_title("(A) forward: RANKL:OPG & outcome vs $B$", fontsize=11, loc="left")
    ax.grid(True, lw=0.4, alpha=0.4)

    # (B) posterior over B from 1 vs 3 measurements
    ax = axes[1]
    for n, col in [(1, "#e67e22"), (3, "#c0392b")]:
        ax.plot(Bgrid, posterior(n), color=col, lw=2, label=f"{n} PICF measurement{'s' if n>1 else ''}")
    ax.axvline(B_true, color="k", ls="--", lw=1.2, label="true $B$")
    ax.set_xlabel("dysbiosis drive $B$"); ax.set_ylabel("posterior density")
    ax.set_title("(B) more data → tighter inference", fontsize=11, loc="left")
    ax.legend(fontsize=8); ax.grid(True, lw=0.4, alpha=0.4)

    # (C) propagate to the risk (36-mo loss) posterior
    ax = axes[2]
    for n, col in [(1, "#e67e22"), (3, "#c0392b")]:
        post = posterior(n)
        # resample B ~ posterior, map to loss
        bs = RNG.choice(Bgrid, size=4000, p=post / post.sum())
        ls = np.interp(bs, Bgrid, loss_map)
        ax.hist(ls, bins=30, density=True, color=col, alpha=0.5, label=f"{n} meas. (CI95 {np.percentile(ls,2.5):.1f}–{np.percentile(ls,97.5):.1f})")
    ax.axvline(loss_true, color="k", ls="--", lw=1.2, label=f"true = {loss_true:.1f} mm")
    ax.set_xlabel("predicted 36-mo bone loss [mm]"); ax.set_ylabel("posterior density")
    ax.set_title("(C) risk estimate tightens (VoI)", fontsize=11, loc="left")
    ax.legend(fontsize=7.5); ax.grid(True, lw=0.4, alpha=0.4)

    # (D) cohort calibration: infer risk for every real patient from 3 PICF readings, vs the truth
    inferred, truth, lo, hi = [], [], [], []
    for b in Bdata:
        rt, lt = forward(b, A)
        o = rt + sigma * RNG.standard_normal(3)
        ll = -0.5 * ((obs_map[None, :] - o[:, None]) / sigma) ** 2
        post = np.exp(ll.sum(0) - ll.sum(0).max()); post /= post.sum()
        bs = RNG.choice(Bgrid, size=3000, p=post); ls = np.interp(bs, Bgrid, loss_map)
        inferred.append(np.median(ls)); truth.append(lt)
        lo.append(np.percentile(ls, 2.5)); hi.append(np.percentile(ls, 97.5))
    inferred = np.array(inferred); truth = np.array(truth)
    axes[3].errorbar(truth, inferred, yerr=[inferred - np.array(lo), np.array(hi) - inferred],
                     fmt="o", color="#c0392b", ecolor="0.7", ms=6, capsize=2)
    lim = [0, max(truth.max(), inferred.max()) * 1.05]
    axes[3].plot(lim, lim, "k--", lw=1.0, label="ideal $y=x$")
    axes[3].axhline(LOSS_OK_C := 2.0, color="0.6", ls=":", lw=0.8); axes[3].axvline(2.0, color="0.6", ls=":", lw=0.8)
    acc = float(np.mean((inferred > 2.0) == (truth > 2.0)))
    from scipy.stats import pearsonr
    r = pearsonr(truth, inferred)[0]
    axes[3].set_xlabel("true 36-mo loss [mm]"); axes[3].set_ylabel("inferred (median ± CI95) [mm]")
    axes[3].set_title(f"(D) cohort calibration\n$r$={r:.2f}, >2 mm acc={acc:.0%} (n={len(truth)})",
                      fontsize=10, loc="left")
    axes[3].legend(fontsize=8); axes[3].grid(True, lw=0.4, alpha=0.4)
    print(f"cohort calibration: Pearson r={r:.2f}, risk-class (>2mm) accuracy={acc:.0%}")

    fig.suptitle("PICF inverse problem: RANKL:OPG → posterior peri-implantitis risk (model)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_picf_inverse.{ext}", dpi=200)

    for n in (1, 3):
        post = posterior(n); bs = RNG.choice(Bgrid, size=8000, p=post / post.sum())
        ls = np.interp(bs, Bgrid, loss_map)
        print(f"{n} PICF meas: risk 36-mo loss = {np.median(ls):.2f} mm  CI95 [{np.percentile(ls,2.5):.2f}, {np.percentile(ls,97.5):.2f}]  (true {loss_true:.2f})")
    print(f"wrote {(OUT / 'fig_picf_inverse.pdf').name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
