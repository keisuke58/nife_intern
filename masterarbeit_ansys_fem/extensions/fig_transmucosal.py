"""Anatomically faithful peri-implant transmucosal model: where the biofilm actually sits.
Three stages (healthy sulcus -> moderate pocket -> deep infrabony pocket) from the axisymmetric CAX4
solves (FEM/tier2b_real/tm_{healthy,moderate,deep}_field.json). The biofilm is the thin RED layer ON
the titanium WITHIN the gingival sulcus -- between the Ti and the peri-implant mucosa -- descending
apically along the implant and into the resorbed (infrabony) bone defect as peri-implantitis advances.
Mirrored about the implant axis for an anatomical view.
Run: python masterarbeit_ansys_fem/extensions/fig_transmucosal.py
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/home/nishioka/IKM_Hiwi/nife"); sys.path.insert(0, str(ROOT))
from thesis_style import use  # noqa: E402
FEM = Path("/home/nishioka/IKM_Hiwi/FEM/tier2b_real")
OUT = ROOT / "masterarbeit_ansys_fem" / "figures"
COL = {"TI": "#636363", "BONE": "#e7d8b0", "GINGIVA": "#f2a0ad", "BIOFILM": "#d62728"}
STAGES = [("healthy", "healthy sulcus"), ("moderate", "peri-mucositis / early"),
          ("deep", "peri-implantitis (infrabony)")]


def main():
    use()
    fig, axes = plt.subplots(1, 3, figsize=use(width_frac=1.0, aspect=0.5), sharey=True)
    for ax, (tag, title) in zip(axes, STAGES):
        d = json.load(open(FEM / ("tm_%s_field.json" % tag)))
        r = np.array([e["r"] for e in d]); z = np.array([e["z"] for e in d])
        mat = np.array([e["mat"] for e in d])
        for rr in (r, -r):                                   # mirror about axis
            for m, c in COL.items():
                s = mat == m
                ax.scatter(rr[s], z[s], c=c, s=7, marker="s", linewidths=0)
        ax.axhline(0, color="0.45", lw=0.7, ls="--")
        ax.set_title(title, fontsize=7.0)
        ax.set_xlim(-5.2, 5.2); ax.set_ylim(-8.2, 4.2); ax.set_aspect("equal")
        ax.set_xlabel(r"radius $r$ (mm)", fontsize=6.8); ax.tick_params(labelsize=6)
    axes[0].set_ylabel(r"axial $z$ (mm), crest $=0$", fontsize=6.8)
    # annotations on the last panel
    axes[2].text(0, 3.7, "abutment (Ti)", ha="center", fontsize=5.6, color="0.3")
    axes[2].text(4.2, 1.6, "gingiva", ha="center", fontsize=5.6, color="#b03050")
    axes[2].text(0.2, -3.2, "bone", ha="center", fontsize=5.6, color="0.45")
    axes[2].annotate("biofilm on Ti,\ndescending into\nthe bony defect", xy=(2.4, -1.6),
                     xytext=(3.0, -6.2), fontsize=5.6, color="#d62728",
                     arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.7))
    handles = [plt.Line2D([], [], marker="s", ls="", mfc=COL[k], mec="none",
                          label={"TI": "implant/abutment", "GINGIVA": "gingiva (mucosa)"}.get(k, k.title()))
               for k in ["TI", "BONE", "GINGIVA", "BIOFILM"]]
    axes[0].legend(handles=handles, fontsize=5.0, loc="lower left", framealpha=0.9, handletextpad=0.2)
    fig.suptitle(r"Peri-implant biofilm location: on the titanium, in the sulcus between Ti and gingiva, "
                 r"descending to bone", fontsize=7.3, y=1.0)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fem_transmucosal.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "fem_transmucosal.pdf")


if __name__ == "__main__":
    main()
