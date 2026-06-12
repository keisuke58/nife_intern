#!/usr/bin/env bash
# Mesh-convergence study of the residual-stress column (verification, item ②).
# Refines the through-thickness mesh N = 4,8,16,32; the depth-graded growth (same CLSM phi_Pg(z)
# server, height-normalized) should converge the peak |S11| and the top-layer S11.
# Run in a real terminal (Abaqus needed):  abaqus/run_convergence.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; PROTO="$(dirname "$HERE")"
PORT="${BRIDGE_PORT:-50007}"; export BRIDGE_HOST=127.0.0.1 BRIDGE_PORT="$PORT"
NS="${*:-4 8 16 32}"

echo "[conv] starting column server"
pkill -f umat_server 2>/dev/null || true
for i in $(seq 1 10); do python -c "import socket;socket.socket().bind(('127.0.0.1',$PORT))" 2>/dev/null && break; sleep 1; done
( cd "$PROTO" && exec python umat_server_col.py --port "$PORT" --profile phipg_depth_DH.json --gamma 0.4 --nramp 20 ) &
trap 'pkill -f umat_server 2>/dev/null || true' EXIT
for i in $(seq 1 20); do python -c "import socket;socket.create_connection(('127.0.0.1',$PORT),0.3).close()" 2>/dev/null && break; sleep 1; done

cd "$HERE"
echo "N  peak|S11|  S11_top" | tee convergence.txt
for N in $NS; do
  python gen_column_inp.py "$N" "column_c$N.inp" >/dev/null
  rm -f "column_c$N".lck 2>/dev/null || true
  abaqus job="column_c$N" user=umat_socket_col.f interactive ask_delete=OFF >/dev/null 2>&1
  if [ -f "column_c$N.odb" ]; then
    abaqus python extract_col_peak.py "column_c$N" | tee -a convergence.txt
  else
    echo "$N  FAILED (see column_c$N.dat)" | tee -a convergence.txt
  fi
done
echo "[conv] done -> convergence.txt  (peak|S11| and S11_top should plateau as N grows)"
