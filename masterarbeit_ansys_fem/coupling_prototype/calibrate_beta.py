"""Calibrate the growth-coupling beta from CLSM z-profiles (physical grounding).

Data: results/diffusion_fit/zprofiles_all_ti.csv — per (condition, day), 5-species occupancy vs
physical depth z_um. We extract:
  * biofilm thickness  h  = max z_um with total occupancy > thr   (substratum-normal extent)
  * Pg fraction        phi_Pg = depth-integrated P.gingivalis / total occupancy

Physical model: in the HOBIC flow chamber the lateral extent (FOV) is fixed, so biofilm thickening
is essentially uniaxial in z. The growth eigenstrain is therefore ANISOTROPIC (depth direction):
    eps_zz_growth = lambda_z - 1,   lambda_z = h(t) / h_ref
Regress eps_zz_growth against phi_Pg over the dysbiotic (DH) series (Pg-driven) -> beta_depth.
CH is the commensal control. Writes beta_calibration.json for nsp_mechanics_model.

Run:  python calibrate_beta.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

NIFE_ROOT = Path('/home/nishioka/IKM_Hiwi/nife')
CSV = NIFE_ROOT / 'results' / 'diffusion_fit' / 'zprofiles_all_ti.csv'
OUT = Path(__file__).resolve().parent / 'beta_calibration.json'
SP = ['S.oralis', 'A.naeslundii', 'Vd/Vp', 'F.nucleatum', 'P.gingivalis']
OCC_THR = 0.05


def thickness_and_pg(df):
    rows = []
    for (c, d), g in df.groupby(['condition', 'day']):
        g = g.sort_values('z_um')
        tot = g[SP].sum(axis=1).values
        z = g.z_um.values
        h = float(z[tot > OCC_THR].max()) if (tot > OCC_THR).any() else 0.0
        pg = float(g['P.gingivalis'].sum() / g[SP].sum().sum())
        rows.append((c, int(d), h, pg))
    return pd.DataFrame(rows, columns=['cond', 'day', 'h_um', 'phi_Pg'])


def main():
    df = pd.read_csv(CSV)
    r = thickness_and_pg(df)
    print(r.to_string(index=False))

    out = {"source": str(CSV.relative_to(NIFE_ROOT)), "occ_threshold": OCC_THR,
           "model": "anisotropic depth growth: eps_zz = beta_depth*phi_Pg (+intercept)"}
    for cond in ['DH', 'CH']:
        m = r[r.cond == cond].sort_values('day')
        if len(m) < 2:
            continue
        h_ref = m.h_um.min()
        eps_zz = (m.h_um.values - h_ref) / h_ref       # depth growth strain vs thinnest slice
        phi = m.phi_Pg.values
        beta, intcpt = np.polyfit(phi, eps_zz, 1)
        corr = float(np.corrcoef(phi, eps_zz)[0, 1])
        out[cond] = {"h_ref_um": float(h_ref), "beta_depth": float(beta),
                     "intercept": float(intcpt), "corr": corr,
                     "phi_Pg_range": [float(phi.min()), float(phi.max())],
                     "h_um_range": [float(m.h_um.min()), float(m.h_um.max())]}
        print(f"\n[{cond}] eps_zz = {beta:.2f}*phi_Pg + {intcpt:.2f}   "
              f"corr={corr:.2f}   (thickness {m.h_um.min():.0f}->{m.h_um.max():.0f} um)")

    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT.name}: beta_depth(DH)={out.get('DH', {}).get('beta_depth', float('nan')):.2f}")
    print("NOTE: lambda_z up to ~3.5 is LARGE-strain growth — small-strain beta is a linearisation;"
          " the thesis should use a multiplicative growth split F = F_e F_g, F_g=diag(1,1,lambda_z).")


if __name__ == "__main__":
    main()
