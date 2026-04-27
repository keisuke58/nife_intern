#!/usr/bin/env python3
"""
Download the ~10 AGORA2 XML models needed for oral guild sign validation.

AGORA2 models are hosted at VMH (Virtual Metabolic Human):
  https://www.vmh.life/files/reconstructions/AGORA2/

Each model is a ~1-4 MB SBML XML file.
Run this script once; it saves models to nife/data/agora2_xml/.

Usage:
  python download_agora2_oral.py [--out_dir /path/to/agora2_xml]
"""
import argparse, urllib.request, hashlib
from pathlib import Path

# (organism_name, filename_at_VMH)
# Verified filenames from AGORA2 v2.01 release
ORAL_MODELS = [
    # Guild                        VMH filename (no extension shown, .xml assumed)
    ('Actinobacteria',   'Actinomyces_naeslundii_MG_1'),
    ('Coriobacteriia',   'Atopobium_parvulum_DSM_20469'),
    ('Bacilli',          'Streptococcus_gordonii_str_Challis_substr_CH1'),
    ('Clostridia',       'Parvimonas_micra_ATCC_33270'),
    ('Negativicutes',    'Veillonella_parvula_DSM_2462'),
    ('Bacteroidia',      'Prevotella_melaninogenica_ATCC_25845'),
    ('Flavobacteriia',   'Capnocytophaga_gingivalis_ATCC_33624'),
    ('Fusobacteriia',    'Fusobacterium_nucleatum_subsp_nucleatum_ATCC_25586'),
    ('Betaproteobacteria','Eikenella_corrodens_ATCC_23834'),
    ('Gammaproteobacteria','Haemophilus_parainfluenzae_T3T1'),
]

BASE_URL = 'https://www.vmh.life/files/reconstructions/AGORA2/reconstructions/mat/{name}.mat'
# AGORA2 also provides SBML:
SBML_URL = 'https://www.vmh.life/files/reconstructions/AGORA2/reconstructions/sbml/{name}.xml'


def download_one(name: str, out_dir: Path) -> Path | None:
    dest = out_dir / f'{name}.xml'
    if dest.exists():
        print(f'  Already downloaded: {dest.name}')
        return dest
    url = SBML_URL.format(name=name)
    print(f'  Downloading {name} ...', end=' ', flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        size_kb = dest.stat().st_size // 1024
        print(f'OK ({size_kb} KB)')
        return dest
    except Exception as e:
        print(f'FAILED: {e}')
        if dest.exists():
            dest.unlink()
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='data/agora2_xml')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Saving AGORA2 oral models to: {out_dir}')
    print(f'Downloading {len(ORAL_MODELS)} models ...')
    ok, fail = [], []
    for guild, name in ORAL_MODELS:
        p = download_one(name, out_dir)
        (ok if p else fail).append((guild, name))

    print(f'\n{len(ok)}/{len(ORAL_MODELS)} downloaded successfully.')
    if fail:
        print('Failed:')
        for g, n in fail:
            print(f'  {g}: {n}')
        print('\nManual download: go to vmh.life → AGORA2 → download individual XML files')
    else:
        print('All models ready. Run:')
        print(f'  python guild_agora_signs.py --agora_dir {out_dir} --plot')
