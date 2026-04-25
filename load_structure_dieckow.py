#!/usr/bin/env python3
"""
Load Dieckow CLSM structural data and build phi rescaling weights.

Structural data: Abutment_Structure vs composition.xlsx (Sheet1)
  VolumeLive [µm³/µm² normalised]
  TotalArea  [µm²]
  PerLive    [%]

Rescaled phi:
  occupancy_i = VolumeLive_i / TotalArea_i   (live biovolume density)
  occupancy_norm_i = occupancy_i / max(occupancy)   ∈ (0,1]
  phi_rescaled[p,w,g] = phi_16S[p,w,g] * occupancy_norm[p,w]
  phi0[p,w] = 1 - occupancy_norm[p,w]   (free resource fraction)
"""

from pathlib import Path
import numpy as np
import pandas as pd

_STRUCT_XLSX = (
    Path(__file__).resolve().parent
    / 'Datasets'
    / 'Abutment_Structure vs composition.xlsx'
)

_PATIENT_LABELS = list('ABCDEFGHIJKL')   # all 12 original patients


def load_structural_data(xlsx_path=None):
    """Return dict of {variable: {(patient, week): float}}."""
    path = Path(xlsx_path) if xlsx_path else _STRUCT_XLSX
    df = pd.read_excel(path, sheet_name='Sheet1', header=None)

    pw_cols = [
        (str(df.iloc[1, c]).strip(), int(df.iloc[2, c]))
        for c in range(1, df.shape[1])
        if pd.notna(df.iloc[1, c]) and pd.notna(df.iloc[2, c])
    ]

    data = {}
    for r in range(3, df.shape[0]):
        label = df.iloc[r, 0]
        if pd.isna(label):
            continue
        label = str(label).strip()
        vals = {}
        for c_idx, (pat, wk) in enumerate(pw_cols):
            v = df.iloc[r, c_idx + 1]
            if pd.notna(v):
                vals[(pat, wk)] = float(v)
        data[label] = vals

    return data


def build_occupancy(struct_data=None, *, normalize=True):
    """
    Compute live-biovolume occupancy = VolumeLive / TotalArea per (patient, week).

    Returns
    -------
    occ : dict  {(patient, week): float}
    max_occ : float   (denominator used for normalization)
    """
    if struct_data is None:
        struct_data = load_structural_data()

    vl = struct_data['VolumeLive']
    ta = struct_data['TotalArea']

    occ = {}
    for key in vl:
        area = ta.get(key, 0.0)
        if area > 1e-9:
            occ[key] = vl[key] / area
        else:
            occ[key] = 0.0

    max_occ = max(occ.values()) if occ else 1.0
    if normalize and max_occ > 0:
        occ = {k: v / max_occ for k, v in occ.items()}

    return occ, max_occ


def rescale_phi(phi_obs, patients, n_weeks=3, xlsx_path=None):
    """
    Rescale phi_obs by live-biovolume occupancy.

    Parameters
    ----------
    phi_obs : np.ndarray  (n_patients, n_weeks, n_guilds)
    patients : list of str  patient labels matching phi_obs rows
    n_weeks : int

    Returns
    -------
    phi_rescaled : np.ndarray  same shape — φ_i * occupancy_norm
    phi0_arr    : np.ndarray  (n_patients, n_weeks) — free resource fraction
    occ_arr     : np.ndarray  (n_patients, n_weeks) — occupancy_norm values
    """
    struct_data = load_structural_data(xlsx_path)
    occ, max_occ = build_occupancy(struct_data, normalize=True)
    per_live = struct_data.get('PerLive', {})

    n_p, n_w, n_g = phi_obs.shape
    occ_arr   = np.ones((n_p, n_w))
    perlive_arr = np.ones((n_p, n_w))

    for p_idx, pat in enumerate(patients):
        for w in range(n_w):
            wk = w + 1
            key = (pat, wk)
            if key in occ:
                occ_arr[p_idx, w] = occ[key]
            if key in per_live:
                perlive_arr[p_idx, w] = per_live[key] / 100.0

    phi_rescaled = phi_obs * occ_arr[:, :, np.newaxis]
    phi0_arr     = 1.0 - occ_arr                        # free resource

    print(f'Occupancy (VolumeLive/TotalArea, normalised):')
    for p_idx, pat in enumerate(patients):
        row = [f'{occ_arr[p_idx,w]:.3f}' for w in range(n_w)]
        print(f'  {pat}: ' + '  '.join(row))
    print(f'  (max_occ = {max_occ:.4f} µm³/µm²)')

    return phi_rescaled, phi0_arr, occ_arr, perlive_arr


if __name__ == '__main__':
    import sys, json
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    phi_raw = np.load(
        Path(__file__).parent / 'results/dieckow_otu/phi_guild_excel_class.npy'
    )
    s = json.load(open(Path(__file__).parent / 'results/dieckow_otu/guild_summary_excel_class.json'))
    patients = list('ABCDEFGHIJKL')[:phi_raw.shape[0]]

    phi_r, phi0, occ, pl = rescale_phi(phi_raw, patients)
    print(f'\nphi_obs  sum (before rescale): {phi_raw.sum(axis=2).mean():.4f}')
    print(f'phi_rescaled sum (after):      {phi_r.sum(axis=2).mean():.4f}')
    print(f'phi0 mean:                     {phi0.mean():.4f}')
