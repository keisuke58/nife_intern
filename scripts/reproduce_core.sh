#!/usr/bin/env bash
# Regenerate LIGHT thesis figures + Dieckow paper figures + run ETL tests.
# No cluster, no GPU, no FISH decode, no AGORA/FBA stack.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> [1/3] LIGHT data figures (make figures)"
make figures

echo "==> [2/3] Dieckow paper figures (dieckow_paper/make_figures.py)"
if [[ -d results/dieckow_cr ]]; then
  python dieckow_paper/make_figures.py
else
  echo "    SKIP: results/dieckow_cr not found (run LOO fits first)"
fi

echo "==> [3/3] ETL unit tests"
pip install -q -e ".[dev]" 2>/dev/null || pip install -q -r requirements-core.txt
python -m pytest tests/ -q --tb=line

echo "==> Done. See docs/PROVENANCE.md for figure map."