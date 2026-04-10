"""Pipeline overview figure: NGS → GEM → dFBA → COMETS (oral biofilm)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os

C0  = "#2E86AB"
C1  = "#A23B72"
C2  = "#F18F01"
C3  = "#1B998B"
C4  = "#E84855"
CBG = "#F4F7FA"

fig, ax = plt.subplots(figsize=(13, 6.5))
fig.patch.set_facecolor(CBG)
ax.set_facecolor(CBG)
ax.set_xlim(0, 13)
ax.set_ylim(0, 6.5)
ax.axis("off")

# ── helpers ──────────────────────────────────────────────────────────────────
def rbox(cx, cy, w, h, color, text, fs=9):
    x, y = cx - w/2, cy - h/2
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.1", linewidth=0, facecolor=color, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fs, fontweight="bold", color="white", zorder=4,
            linespacing=1.4)

def sub(cx, cy, text, fs=7):
    ax.text(cx, cy, text, ha="center", va="top",
            fontsize=fs, color="#666666", linespacing=1.3)

def harrow(x0, x1, y, color, lw=1.5):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        mutation_scale=15, shrinkA=4, shrinkB=4))

def varrow(x, y0, y1, color, lw=1.5):
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        mutation_scale=15, shrinkA=4, shrinkB=4))

def alabel(cx, cy, text, fs=7):
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fs, color="#555555", linespacing=1.2)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(6.5, 6.15, "NGS  →  GEM  →  dFBA  →  COMETS   oral biofilm pipeline",
        ha="center", va="center", fontsize=11.5, fontweight="bold", color="#222222")

# ─────────────────────────────────────────────────────────────────────────────
# ROW 1  (y=5.0):  NGS  →  MetaPhlAn  →  init_comp
# ─────────────────────────────────────────────────────────────────────────────
rbox(0.9,  5.0, 1.4, 0.7, C0,       "NGS\n(Shotgun)")
rbox(2.95, 5.0, 1.5, 0.7, "#1D6E8C", "MetaPhlAn 4")
rbox(5.1,  5.0, 1.8, 0.7, "#3A8F8A", "init_comp.json")

sub(0.9,  4.62, "Patient FASTQ")
sub(2.95, 4.62, "% abundance / genus")
sub(5.1,  4.62, "So / An / Vp / Fn / Pg")

harrow(1.61, 2.19, 5.0, C0)                   # NGS→MetaPhlAn
alabel(1.9,  5.2,  "bowtie2", fs=6.5)

harrow(3.71, 4.19, 5.0, "#1D6E8C")             # MetaPhlAn→init_comp
alabel(3.95, 5.22, "metaphlan_feature_table_to_init_comp.py", fs=6)

# ─────────────────────────────────────────────────────────────────────────────
# COMETS bounding box  (x: 7.15..12.7,  y: 1.8..6.1)
# ─────────────────────────────────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch((7.15, 1.85), 5.35, 4.0,
    boxstyle="round,pad=0.1", linewidth=2,
    edgecolor=C3, facecolor=C3+"12", zorder=1))
ax.text(9.82, 2.06, "COMETS framework  (Java + cometspy)",
        ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=C3)

# ── AGORA  (y=5.55, dashed) ───────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch((7.65, 5.2), 4.35, 0.75,
    boxstyle="round,pad=0.07", linewidth=1.4, linestyle="dashed",
    edgecolor=C1, facecolor=C1+"18", zorder=2))
ax.text(9.82, 5.575, "AGORA  GEM library",
        ha="center", va="center", fontsize=8.5, fontweight="bold", color=C1)
sub(9.82, 5.21, "~800 reconstructions  ·  oral / gut microbiome", fs=6.8)

# ── GEM  (y=4.1) ──────────────────────────────────────────────────────────────
rbox(9.82, 4.1, 2.2, 0.72, C1, "GEM\n(per species)")
sub(9.82, 3.72, "~1000 rxns  /  ~800 metabolites")

varrow(9.82, 5.2, 4.47, C1)                   # AGORA→GEM
alabel(10.5, 4.83, "select 5 sp.", fs=6.5)

# init_comp → GEM
ax.annotate("", xy=(8.72, 4.2), xytext=(6.0, 5.0),
    arrowprops=dict(arrowstyle="-|>", color="#3A8F8A", lw=1.4,
                    mutation_scale=14, shrinkA=4, shrinkB=4))
alabel(7.5, 4.75, "initial fractions", fs=6.5)

# ── dFBA  (y=2.9) ─────────────────────────────────────────────────────────────
rbox(9.82, 2.9, 2.2, 0.72, C2, "dFBA")
sub(9.82, 2.52, "LP / species / Δt  ·  max μ  s.t. stoich.+med.")

varrow(9.82, 3.74, 3.26, C1)                  # GEM→dFBA
alabel(10.6, 3.5, "metabolic\nconstraints", fs=6.5)

# medium-update loop (right side)
ax.annotate("", xy=(11.3, 3.74), xytext=(11.3, 3.26),
    arrowprops=dict(arrowstyle="-|>", color=C2, lw=1.3,
                    mutation_scale=13, connectionstyle="arc3,rad=-0.6"))
alabel(12.1, 3.5, "medium\nupdate\nt→t+Δt", fs=6.5)

# ─────────────────────────────────────────────────────────────────────────────
# Metabolic profile  (y=2.9, left of COMETS)
# ─────────────────────────────────────────────────────────────────────────────
rbox(3.5, 2.9, 2.4, 0.72, C4, "Metabolic profile")
sub(3.5, 2.52, "lactate · succinate · pH proxy  (80 h)")

harrow(8.71, 4.71, 2.9, C2)                   # dFBA→metabolic profile
alabel(6.7, 3.08, "80 h sim", fs=6.5)

# ─────────────────────────────────────────────────────────────────────────────
# Cross-feeding  (y=1.6)
# ─────────────────────────────────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch((1.5, 1.4), 4.0, 0.58,
    boxstyle="round,pad=0.07", linewidth=1,
    edgecolor=C2+"99", facecolor=C2+"10", zorder=2))
ax.text(3.5, 1.69,
        "So/An  →(lactate)→  Vp/Fn        An  →(succinate)→  Pg",
        ha="center", va="center", fontsize=7.5, color="#444444")
ax.annotate("", xy=(3.5, 2.17), xytext=(3.5, 1.98),
    arrowprops=dict(arrowstyle="-|>", color=C2+"99", lw=0.9, mutation_scale=9))

# ── Legend ────────────────────────────────────────────────────────────────────
ax.legend(handles=[
    mpatches.Patch(color=C0,  label="Data"),
    mpatches.Patch(color=C1,  label="GEM / AGORA"),
    mpatches.Patch(color=C2,  label="dFBA algorithm"),
    mpatches.Patch(color=C3,  label="COMETS framework"),
    mpatches.Patch(color=C4,  label="Output"),
], loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=5,
   fontsize=7.5, framealpha=0.0, handlelength=1.2, columnspacing=1.2)

plt.tight_layout(pad=0.3)
os.makedirs("pipeline_results", exist_ok=True)
plt.savefig("pipeline_results/pipeline_overview.png",
            dpi=180, bbox_inches="tight", facecolor=CBG)
print("Saved: pipeline_results/pipeline_overview.png")
