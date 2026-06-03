#!/usr/bin/env python3
"""
aggregate_diffusion_sweep.py — collect the diffusion-fit sweep results.

Scans results/diffusion_fit/D_fit_*.json (the canonical untagged CH/DH plus the
tagged sweep runs), tabulates convergence / loss / fitted D, u and the
hyper-parameters, and writes a summary CSV. Used to pick the best setting after
the PBS sweep (fit_diffusion_sweep_submit.sh) finishes.

Run from the repo root:
    python scripts/analysis/aggregate_diffusion_sweep.py
"""
from __future__ import annotations
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[2]
FITDIR = HERE / 'results' / 'diffusion_fit'
SHORT = ['S.o', 'A.n', 'Vd/Vp', 'F.n', 'P.g']


def main():
    rows = []
    for jf in sorted(FITDIR.glob('D_fit_*.json')):
        d = json.loads(jf.read_text())
        hp = d.get('hparams', {})
        row = {
            'cond': d.get('cond'),
            'tag': d.get('tag', '') or '(baseline)',
            'success': d.get('success'),
            'loss': d.get('loss'),
            'u': d.get('u_fit'),
            'Nz': hp.get('Nz'), 'dt': hp.get('dt'), 'fd_eps': hp.get('fd_eps'),
            'restarts': hp.get('restarts'),
        }
        for k, sp in enumerate(SHORT):
            row[f'D[{sp}]'] = d['D_fit'][k] if d.get('D_fit') and k < len(d['D_fit']) else None
        row['file'] = jf.name
        rows.append(row)

    if not rows:
        print('no D_fit_*.json found in', FITDIR)
        return
    df = pd.DataFrame(rows).sort_values(['cond', 'loss'], na_position='last')
    pd.set_option('display.width', 200, 'display.max_columns', 30)
    print(df.drop(columns=['file']).to_string(index=False))

    out = FITDIR / 'sweep_summary.csv'
    df.to_csv(out, index=False)
    print(f'\nwrote {out}  ({len(df)} runs)')

    # best (lowest loss) per condition, preferring converged runs
    print('\n=== best per condition (converged first, then lowest loss) ===')
    for cond, g in df.groupby('cond'):
        g2 = g.sort_values(['success', 'loss'], ascending=[False, True])
        b = g2.iloc[0]
        print(f"  {cond}: tag={b['tag']}  success={b['success']}  loss={b['loss']:.5f}  "
              f"Nz={b['Nz']} fd_eps={b['fd_eps']}")


if __name__ == '__main__':
    main()
