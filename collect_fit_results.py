#!/usr/bin/env python3
"""
Collect and compare all guild fit results.
Run locally after jobs finish to get final RMSE table and update paper numbers.

Usage:
  python3 collect_fit_results.py [--watch]   # --watch polls every 60s until jobs done
"""

import json, sys, time, subprocess
import numpy as np
from pathlib import Path

CR  = Path(__file__).resolve().parent / 'results' / 'dieckow_cr'
OTU = Path(__file__).resolve().parent / 'results' / 'dieckow_otu'

FITS = {
    'gLV (L-BFGS-B, excel_class)': CR / 'fit_guild_excel_class.json',
    'gLV (L-BFGS-B, full)':        CR / 'fit_guild.json',
    'Hamilton (JAX-LBFGS)':        CR / 'fit_guild_jaxlbfgs_best.json',
    'Hamilton (Adam, baseline)':    CR / 'fit_guild_hamilton.json',
    'Hamilton (Adam, masked+struct, CPU)':  CR / 'fit_guild_hamilton_masked_struct.json',
    'Hamilton (Adam, masked+struct, GPU)':  CR / 'fit_guild_hamilton_masked_struct_gpu.json',
    # legacy / alternate names
    'Hamilton (Adam, masked CPU)':  CR / 'fit_guild_hamilton_masked.json',
    'Hamilton (Adam, masked GPU)':  CR / 'fit_guild_hamilton_masked_gpu.json',
}

LOGS = {
    'CPU': Path(__file__).resolve().parent / 'results' / 'dieckow_cr' / 'fit_guild_hamilton_masked.log',
    'GPU': Path(__file__).resolve().parent / 'results' / 'dieckow_cr' / 'fit_guild_hamilton_masked.gpu.log',
}


REMOTE_HOST = 'vancouver01'


def _read_remote_log(remote_path, n=20):
    try:
        out = subprocess.check_output(
            ['ssh', REMOTE_HOST, f'tail -{n} {remote_path}'],
            text=True, timeout=15, stderr=subprocess.DEVNULL
        )
        return out.strip().splitlines()
    except Exception:
        return []


def read_current_epoch(log_path):
    """Parse last epoch/loss line; reads from vancouver01 if not local."""
    if log_path.exists():
        lines = log_path.read_text().strip().splitlines()
    else:
        lines = _read_remote_log(str(log_path))
    for line in reversed(lines):
        if 'epoch' in line and 'loss=' in line:
            parts = line.split()
            epoch = loss = None
            for i, p in enumerate(parts):
                if p == 'epoch':
                    try: epoch = int(parts[i+1])
                    except: pass
                if p.startswith('loss='):
                    try: loss = float(p.split('=')[1])
                    except: pass
            return epoch, loss
    return None, None


def print_table():
    print('\n' + '='*72)
    print(f'{"Model":<42} {"RMSE":>8}  {"Notes"}')
    print('-'*72)
    for name, path in FITS.items():
        if not path.exists():
            continue
        d = json.load(open(path))
        rmse = d.get('rmse', float('nan'))
        patients = len(d.get('b_all', []))
        guilds   = len(d.get('guilds', []))
        msg = d.get('message', '')[:30]
        print(f'  {name:<40} {rmse:>8.5f}  (P={patients}, G={guilds})')
    print('-'*72)

    print('\nRunning jobs (latest epoch):')
    for tag, lp in LOGS.items():
        ep, loss = read_current_epoch(lp)
        if ep is not None:
            print(f'  {tag}: epoch {ep}/3000  loss={loss:.5f}')
        else:
            print(f'  {tag}: (no data yet)')
    print()


def jobs_alive():
    try:
        out = subprocess.check_output(
            ['ssh', 'vancouver01',
             'ps aux | grep fit_guild_hamilton_masked | grep -v grep | wc -l'],
            text=True, timeout=15
        ).strip()
        return int(out) > 0
    except Exception:
        return False


def watch_loop():
    while True:
        print_table()
        if not jobs_alive():
            print('Jobs finished.')
            break
        print('Jobs still running. Checking again in 120s... (Ctrl-C to stop)\n')
        time.sleep(120)


if __name__ == '__main__':
    if '--watch' in sys.argv:
        watch_loop()
    else:
        print_table()
