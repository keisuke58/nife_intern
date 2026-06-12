#!/usr/bin/env bash
# A3 patient-specific detachment risk: sweep the growth driver phi over the clinical range,
# run the cohesive-delamination model at each, record delaminated fraction -> risk(phi).
# One server per phi value (the profile changes). tn0 fixed at the calibrated operating point.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; PROTO="$(dirname "$HERE")"
PORT="${BRIDGE_PORT:-50007}"; export BRIDGE_HOST=127.0.0.1 BRIDGE_PORT="$PORT"
UMAT="umat_socket_col.f"; JOB="film_delam"; TN0="${DELAM_TN0:-120}"
PHIS="${PHIS:-0.06 0.10 0.14 0.18 0.22 0.26 0.30}"
GAMMA="${RISK_GAMMA:-0.6}"

cd "$HERE"; python gen_film_delam_inp.py 12 8 "$JOB.inp" "$TN0" 4.0 >/dev/null
OUT="risk_sweep.txt"; : > "$OUT"
for P in $PHIS; do
  printf '{"z_norm":[0.0,1.0],"phi_Pg":[%s,%s]}\n' "$P" "$P" > "$PROTO/phipg_risk.json"
  pkill -f "umat_server" 2>/dev/null || true
  for i in $(seq 1 8); do python -c "import socket;socket.socket().bind(('127.0.0.1',$PORT))" 2>/dev/null && break; sleep 1; done
  ( cd "$PROTO" && exec python umat_server_col.py --profile phipg_risk.json --gamma "$GAMMA" --nramp 40 --port "$PORT" ) &
  SRV=$!
  for i in $(seq 1 30); do python -c "import socket;socket.create_connection(('127.0.0.1',$PORT),0.3).close()" 2>/dev/null && break; sleep 1; done
  rm -f "$JOB".lck 2>/dev/null || true
  abaqus job="$JOB" user="$UMAT" interactive ask_delete=OFF >/dev/null 2>&1 || true
  FRAC=$(abaqus python extract_delam.py "$JOB" 12 8 2>/dev/null | grep -oE "delam length = [0-9.]+  \([0-9]+%" | grep -oE "[0-9]+%" | tr -d '%')
  echo "$P ${FRAC:-NA}" | tee -a "$OUT"
  pkill -f "umat_server" 2>/dev/null || true; kill $SRV 2>/dev/null || true
  sleep 1
done
echo "== phi  delam% =="; cat "$OUT"
