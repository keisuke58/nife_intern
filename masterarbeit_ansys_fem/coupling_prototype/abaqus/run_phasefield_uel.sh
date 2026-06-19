#!/usr/bin/env bash
# Run the AT2 phase-field Abaqus UEL on a single-edge-notched tension (SENT) coupon at two mesh
# densities; checks mesh-independence of the crack length per the standard phase-field test (the
# crack length should be invariant under uniform h-refinement once h ≲ l/2).
#
#   cd /home/nishioka/IKM_Hiwi/FEM/coupling_prototype/abaqus
#   bash run_phasefield_uel.sh
set -euo pipefail
cd "$(dirname "$0")"

ABQ=~/DassaultSystemes/SIMULIA/Commands/abaqus
LX=1.0; LY=1.0
E_MOD=1.0; GC=2.7e-3; ELL=0.025
U_TOP=0.04

# coarse (h ≈ l) and fine (h ≈ l/2); under AT2 the crack length should be insensitive within ~5 %.
for NX in 40 80; do
    NY=$NX
    NAME="phasefield_sent_${NX}"
    echo "=== build $NAME (${NX}x${NY}) ==="
    python gen_phasefield_uel_inp.py "$NAME" "$NX" "$NY" "$LX" "$LY" \
        "$E_MOD" "$GC" "$ELL" "$U_TOP" --notch-band 0.02 --nstep 60

    echo "=== solve $NAME ==="
    rm -f "${NAME}.lck"
    "$ABQ" job="$NAME" user=uel_phasefield_at2.f cpus=1 interactive \
        >"${NAME}.log" 2>&1 || true

    if grep -q COMPLETED "${NAME}.log"; then
        python3 - "$NAME" "$NX" <<'PY'
import sys, re, pathlib, json
job, nx = sys.argv[1], int(sys.argv[2])
dat = pathlib.Path(f"{job}.dat").read_text()
# Count Gauss points whose history H crossed Gc/2 (phenomenological "crack region")
# SDV1 is the history H; SDV2 is the current ψ_+. We pull the last frame block.
blocks = re.findall(r"ELEMENT\s+PT.*?\n(.+?)(?=\n\s*\n|\Z)", dat, re.S)
if not blocks:
    print("no SDV output found"); sys.exit(0)
last = blocks[-1].strip().splitlines()
cracked = sum(1 for L in last
              if (m := re.search(r"\s+([0-9]+\.[0-9eE+-]+)\s+([0-9]+\.[0-9eE+-]+)\s*$", L))
              and float(m.group(1)) > 0.5 * 2.7e-3 * 0.5)   # H > 0.25 Gc (≈ saturation)
n_gp = 4 * nx * nx           # 4 IPs per element × NX^2 elements
rec = {"job": job, "nx": nx, "n_gp": n_gp, "cracked_gp": cracked,
       "cracked_frac": cracked / n_gp}
print(json.dumps(rec))
pathlib.Path("phasefield_uel_results.jsonl").open("a").write(json.dumps(rec) + "\n")
PY
    else
        echo "$NAME FAILED"
        tail -20 "${NAME}.dat" 2>/dev/null || true
    fi
done

echo "=== phase-field UEL sweep done -> phasefield_uel_results.jsonl ==="
python3 - <<'PY'
import json, pathlib
rows = [json.loads(L) for L in
        pathlib.Path("phasefield_uel_results.jsonl").read_text().splitlines() if L.strip()]
if len(rows) >= 2:
    coarse, fine = rows[0], rows[-1]
    rel = abs(fine["cracked_frac"] - coarse["cracked_frac"]) / max(coarse["cracked_frac"], 1e-12)
    print(f"mesh-independence: coarse {coarse['cracked_frac']:.4f}, "
          f"fine {fine['cracked_frac']:.4f}, |Δ|/coarse = {rel*100:.2f} %")
    print("PASS (≤5 %)" if rel <= 0.05 else "FAIL (>5 %) — try smaller l/h")
PY
