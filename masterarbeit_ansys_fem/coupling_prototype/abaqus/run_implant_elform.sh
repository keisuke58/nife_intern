#!/usr/bin/env bash
# Element-FORMULATION robustness matrix for the implant-thread biofilm model.
# Re-solves the SAME geometry / growth eigenstrain at fixed resolution (n108) across element
# formulations to prove the interface peak and (key) the DH/CH & thread/flat RATIOS are not a
# volumetric-locking / root-under-resolution artefact of the headline linear CPE4.
#
# Non-interactive (detached) solves + .lck polling: `abaqus ... interactive` hangs without a TTY,
# so we launch detached and wait on the lock file.  Serial (single shared license).
set -u
export PATH="$HOME/DassaultSystemes/SIMULIA/Commands:$PATH"
cd "$(dirname "$0")"
NX=108; NZ=18; PER=3
OUT=elform_results.csv
echo "eltype,cond,profile,Nx,interior_peak_vm,interior_peak_peel,edge_vm" > "$OUT"

solve() {  # eltype cond profile amp job
  local el="$1" cond="$2" prof="$3" amp="$4" job="$5"
  rm -f "$job".* 2>/dev/null; rm -rf "$job".simdir 2>/dev/null
  python gen_implant_inp_q.py "$cond" "$prof" "$job.inp" "$NX" "$NZ" "$PER" "$amp" "$el" >/dev/null
  # SERIAL ONLY: QSD has 50 tokens; concurrent jobs exhaust them and deadlock. Launch detached, then
  # block on the .sta completion marker (lock-file timing is unreliable under license checkout).
  abaqus job="$job" ask_delete=OFF >/dev/null 2>&1
  local t=0
  while [ $t -lt 300 ]; do
    grep -q "COMPLETED SUCCESSFULLY" "$job.sta" 2>/dev/null && break
    grep -qiE "has not been completed|error" "$job.dat" 2>/dev/null && break
    sleep 3; t=$((t+3))
  done
  if grep -q "COMPLETED SUCCESSFULLY" "$job.sta" 2>/dev/null; then
    abaqus python extract_implant.py "$job" "$NX" >/dev/null 2>&1
    python - "$job" "$el" "$cond" "$prof" "$NX" "$OUT" <<'PY'
import sys, json
job, el, cond, prof, nx, out = sys.argv[1:7]
d = json.load(open(job + "_iface.json"))
with open(out, "a") as f:
    f.write("%s,%s,%s,%s,%.4f,%.4f,%.4f\n" %
            (el, cond, prof, nx, d["interior_peak_vm"], d["interior_peak_peel"], d["peak_vm"]))
print("  %-7s %s/%-6s  interior_vM=%.2f" % (el, cond, prof, d["interior_peak_vm"]))
PY
  else
    echo "  !! FAILED $el $job"
    echo "$el,$cond,$prof,$NX,NaN,NaN,NaN" >> "$OUT"
  fi
}

for EL in CPE4 CPE4H CPE8 CPE8R CPE8RH; do
  echo "=== $EL ==="
  solve "$EL" DH thread 0.18 "ef_${EL}_DH_thr"
  solve "$EL" CH thread 0.18 "ef_${EL}_CH_thr"
  solve "$EL" DH flat   0.0  "ef_${EL}_DH_flat"
  solve "$EL" CH flat   0.0  "ef_${EL}_CH_flat"
done
echo "DONE -> $OUT"
