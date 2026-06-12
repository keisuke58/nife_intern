"""fig_centrality_thesis.py — thesis_style version of the guild-network centrality summary.

Reproduces the content of the old raster `dieckow_centrality.png` (the one figure in the thesis still
in the default matplotlib font) as a vector PDF in the unified thesis_style (usetex/lmodern/9pt).
Influence/vulnerability/net-effect from the canonical fitted A matrix; net-effect shown with
usetex-safe matplotlib triangle markers (up = net mutualist, down = net competitor).

Run from repo root:
  PATH=/home/nishioka/texlive/2025/bin/x86_64-linux:$PATH python scripts/figures/fig_centrality_thesis.py
Output: results/dieckow_centrality.pdf
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import thesis_style
from guild_replicator_dieckow import GUILD_SHORT, GUILD_COLORS, GUILD_ORDER

ROOT = Path(__file__).resolve().parents[2]
FIT = ROOT / "results" / "dieckow_cr" / "fit_guild.json"
OUT = ROOT / "results" / "dieckow_centrality.pdf"

d = json.loads(FIT.read_text())
A = np.array(d["A"])
guilds = d["guilds"]
short = [GUILD_SHORT.get(g, g[:4]) for g in guilds]
colors = [GUILD_COLORS.get(g, "0.5") for g in guilds]
n = len(short)

influence = np.array([np.sum(np.abs(A[:, i])) - np.abs(A[i, i]) for i in range(n)])      # outgoing
vulnerability = np.array([np.sum(np.abs(A[i, :])) - np.abs(A[i, i]) for i in range(n)])   # incoming
net_effect = np.array([np.sum(A[i, :]) - A[i, i] for i in range(n)])
order = np.argsort(influence + vulnerability)[::-1]

fig, ax = plt.subplots(figsize=thesis_style.use(width_frac=1.0, aspect=0.5))
x = np.arange(n); width = 0.38
ax.bar(x - width / 2, influence[order], width, color=[colors[i] for i in order],
       alpha=0.9, label=r"Influence $\Sigma_j|A_{ji}|$ (drives others)")
ax.bar(x + width / 2, vulnerability[order], width, color=[colors[i] for i in order],
       alpha=0.40, edgecolor=[colors[i] for i in order], linewidth=1.0,
       label=r"Vulnerability $\Sigma_j|A_{ij}|$ (affected by others)")
for xi, i in enumerate(order):
    ne = net_effect[i]
    up = ne > 0
    y = max(influence[i], vulnerability[i])
    ax.plot(xi, y + 0.25, marker="^" if up else "v", ms=4,
            color="#2166ac" if up else "#d6604d", clip_on=False)
    ax.text(xi, y + 0.6, r"$%+.1f$" % ne, ha="center", va="bottom", fontsize=6,
            color="#2166ac" if up else "#d6604d")
ax.set_xticks(x); ax.set_xticklabels([short[i] for i in order], rotation=45, ha="right")
ax.set_ylabel(r"$\Sigma|A|$ (off-diagonal)")
ax.set_title(r"Guild network centrality ($\blacktriangle$ net mutualist, $\blacktriangledown$ net competitor)",
             fontsize=8)
ax.legend(fontsize=7, loc="upper right", frameon=False)
ax.axhline(0, color="k", lw=0.5)
ax.set_ylim(0, max(influence.max(), vulnerability.max()) * 1.18)
fig.savefig(OUT, bbox_inches="tight"); plt.close(fig)
print("wrote", OUT)
print("  Actinobacteria outgoing=%.1f net=%+.1f | Bacilli incoming=%.1f"
      % (influence[GUILD_ORDER.index("Actinobacteria")], net_effect[GUILD_ORDER.index("Actinobacteria")],
         vulnerability[GUILD_ORDER.index("Bacilli")]))
