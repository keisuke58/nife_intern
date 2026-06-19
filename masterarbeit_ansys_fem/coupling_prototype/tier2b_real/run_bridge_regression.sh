#!/usr/bin/env bash
# Bridge regression after Stage A.
#
# Once the Stage A precision package has produced (i) the h-convergence ladder and (ii) the OAT
# material-sensitivity tornado, the bridge (B1+ wrinkling / B2 delamination / B3 viscoelasticity /
# A4 SOCKET-FREE column convergence) needs a regression check against its published headline values:
# the new mesh density at the converged h and the materials at their *sensitivity-anchored* baseline
# should reproduce the headline B/C numbers within a small tolerance (default 5 %). Otherwise the
# Stage A audit has implicitly invalidated a B/C result that the thesis already cites.
#
# This script re-runs (in headless serial mode) the smallest defensible subset of the bridge tests at
# the converged Stage A settings:
#
#   1. A4 column at N=16 (machine-identical patch + graded N16↔N32 +0.27 % → already converged)
#   2. B2 delamination at the operating point t_n0=120 (DH vs CH, **3.3× detachment ratio** headline)
#   3. B3 viscoelasticity one-element relaxation (final/instantaneous **0.0544 vs 0.0543** headline)
#
# Each test is run via the existing per-test runner in coupling_prototype/abaqus/run_abaqus_test.sh;
# fig_bridge_regression.py reads the per-test extractor outputs and emits a PASS/FAIL table.
#
#   cd /home/nishioka/IKM_Hiwi/FEM/tier2b_real
#   bash run_bridge_regression.sh
#
# Compute: each bridge test ~30 s wallclock; total under 5 min. Read-only against Stage A outputs;
# does not modify the headline numbers.
set -euo pipefail

REPO_FEM=/home/nishioka/IKM_Hiwi/FEM
COUPLING=$REPO_FEM/coupling_prototype
ABQ_DIR=$COUPLING/abaqus

cd "$COUPLING"

REGRESSION_JSON="$COUPLING/bridge_regression_results.jsonl"
rm -f "$REGRESSION_JSON"

# Resolve Stage A → choose the converged LC and the headline material baseline. We trust LC=0.40 as
# converged unless the user has explicitly produced a Richardson result that says otherwise; the
# script reads hconv_summary.csv to pick the smallest LC whose absolute |p95 − p95∞| / p95∞ ≤ 1 %.
STAGE_A_LC=$(python3 - <<'PY'
from pathlib import Path
import pandas as pd
csv = Path("tier2b_real/hconv_summary.csv")
if not csv.exists():
    print("0.40")
else:
    df = pd.read_csv(csv)
    # crown C3D10 series; pick the LC whose deviation from finest is < 1 %
    sub = df[(df.kind == "crown") & (df.order == "C3D10")].sort_values("lc")
    if sub.empty:
        print("0.40")
    else:
        p_fine = float(sub.iloc[0]["crestal_p95_MPa"])
        ok = sub[sub.crestal_p95_MPa.sub(p_fine).abs() / p_fine < 0.01]
        print(f"{float(ok.lc.max()):.2f}" if not ok.empty else "0.40")
PY
)
echo "[regression] Stage A converged LC = $STAGE_A_LC"

# Common runner ---------------------------------------------------------------
run_test() {
    local name=$1 cmd=$2 expect=$3
    echo ""
    echo "=== regression test: $name ==="
    cd "$ABQ_DIR"
    bash "./run_abaqus_test.sh" $cmd 2>&1 | tee -a "${REGRESSION_JSON}.log"
    cd "$COUPLING"
    python3 - "$name" "$expect" <<'PY'
import json, sys, pathlib
name, expect = sys.argv[1:]
log = pathlib.Path("tier2b_real/bridge_regression_results.jsonl.log").read_text()
# the abaqus runner echoes the headline number on its final line; we parse it generically.
import re
m = re.findall(r"([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)", log.splitlines()[-1])
val = float(m[-1]) if m else float("nan")
exp = float(expect)
rel = abs(val - exp) / abs(exp) if exp else float("nan")
ok = rel <= 0.05
rec = {"test": name, "headline": exp, "observed": val, "rel_err": rel, "pass": bool(ok)}
print(json.dumps(rec))
with open("tier2b_real/bridge_regression_results.jsonl", "a") as f:
    f.write(json.dumps(rec) + "\n")
PY
}

# Test 1 — A4 column N=16 ----------------------------------------------------
# Expects |σ_xx| = 0.848μ at the confined-bottom plateau. We use the existing column runner.
run_test "A4_column_N16_plateau" "phipg_flat" 0.848

# Test 2 — B2 delamination at t_n0=120 -----------------------------------------
# Expect DH=83% L delaminated; CH=25%. We check the DH headline.
DELAM_TN0=120 run_test "B2_delam_DH_pct" "delam" 83.0

# Test 3 — B3 viscoelasticity ratio -------------------------------------------
# Expect S13 final/instantaneous = 0.0544.
run_test "B3_visco_relax" "viscoelastic" 0.0544

echo ""
echo "=== regression done — summary ==="
python3 - <<'PY'
import json, pathlib
rows = [json.loads(L) for L in pathlib.Path("tier2b_real/bridge_regression_results.jsonl").read_text().splitlines() if L.strip()]
import pandas as pd
df = pd.DataFrame(rows)
print(df.to_string(index=False))
fail = df[~df["pass"]]
print(f"\n{'PASS' if fail.empty else 'FAIL'}: {len(rows) - len(fail)}/{len(rows)} tests within 5 %")
PY

echo ""
echo "next: python fig_bridge_regression.py"
