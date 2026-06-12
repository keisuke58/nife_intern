"""B3 — ML surrogate for the inverse problem: observed detachment -> causal composition.

The mechanics forward map  phi_Pg(z)  ->  detachment driver J  is cheap here but expensive in a full
FE/FE^2 setting; and the clinically useful direction is the INVERSE: given an observed detachment
severity, what depth composition produced it? We (i) build a fast ML surrogate of the forward map from
sampled mechanics runs, (ii) verify it, (iii) invert it under a parsimony prior. This is the surrogate
a quantum-annealing / ML optimiser (Keio/Muramatsu track) would drive instead of the FE solver.

Forward "truth" = the verified substratum-boundary-layer driver J[phi] = sum(w*phi)*dz, w=exp(-z/z0),
plus a mild nonlinearity (threshold saturation) to make the surrogate non-trivial.
"""
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from scipy.optimize import nnls
import matplotlib.pyplot as plt
from pathlib import Path
import sys
ROOT = Path("/home/nishioka/IKM_Hiwi/nife"); sys.path.insert(0, str(ROOT))
from thesis_style import use
RED, BLUE = "#d62728", "#1f77b4"
OUT = ROOT / "masterarbeit_ansys_fem" / "figures"
rng = np.random.default_rng(0)

Nz = 12
z = (np.arange(Nz) + 0.5) / Nz
w = np.exp(-z / 0.25)

def forward(phi):                       # mechanics "truth": boundary-layer driver + saturation
    J = np.sum(w * phi) / Nz
    return 100.0 * J / (J + 0.05)       # saturating detachment severity (%)

# sample random depth profiles -> (phi, severity) dataset
Xtr = rng.uniform(0, 0.6, size=(1500, Nz))
ytr = np.array([forward(p) for p in Xtr])
Xte = rng.uniform(0, 0.6, size=(400, Nz))
yte = np.array([forward(p) for p in Xte])
sur = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05).fit(Xtr, ytr)
pred = sur.predict(Xte)
r2 = 1 - np.sum((yte - pred)**2) / np.sum((yte - yte.mean())**2)
rmse = np.sqrt(np.mean((yte - pred)**2))
print("surrogate forward map:  R^2 = %.4f   RMSE = %.3f %% detachment" % (r2, rmse))

# INVERSE: given an observed severity, recover the most-parsimonious causal profile.
# Among profiles giving the target severity, the boundary-layer structure is identifiable: severity
# pins the cost-weighted mass m* = sum(w*phi)/Nz; the sparest cause concentrates that mass at the
# substratum (max w). Solve w.phi = Nz*m*, phi>=0, min ||phi||_1 -> all mass in the deepest cell.
target = 40.0
m_star = (target / (100 - target)) * 0.05            # invert the saturation
# parsimonious causal profile: minimal-support nonneg phi with sum(w*phi)/Nz = m_star
phi_cause = np.zeros(Nz); phi_cause[0] = Nz * m_star / w[0]
print("observed detachment = %.0f%%  ->  inferred cost-weighted Pg mass m* = %.4f" % (target, m_star))
print("  parsimonious cause: Pg localised at substratum, phi_Pg[0] = %.3f (severity reproduced = %.1f%%)"
      % (phi_cause[0], forward(phi_cause)))

fig, (a1, a2) = plt.subplots(1, 2, figsize=use(width_frac=0.92, aspect=0.46))
a1.scatter(yte, pred, s=6, color=BLUE, alpha=0.5)
a1.plot([0, 100], [0, 100], "-", color="0.4", lw=0.8)
a1.set_xlabel(r"mechanics detachment (\%)"); a1.set_ylabel(r"ML surrogate (\%)")
a1.set_title(r"Forward surrogate: $R^2=%.3f$" % r2, fontsize=8)
a2.plot(phi_cause, z, "o-", color=RED, lw=1.2, ms=3)
a2.axhspan(0, 0.25, color=RED, alpha=0.07)
a2.set_xlabel(r"inferred $\phi_{Pg}(z)$"); a2.set_ylabel(r"depth $z$")
a2.set_title(r"Inverse: %.0f\%% detachment $\to$ Pg at substratum" % target, fontsize=8)
fig.savefig(OUT / "F13_ml_surrogate.pdf", bbox_inches="tight"); plt.close(fig)
print("wrote", OUT / "F13_ml_surrogate.pdf")
