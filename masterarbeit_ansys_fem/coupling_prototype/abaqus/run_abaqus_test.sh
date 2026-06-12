#!/usr/bin/env bash
# Single-element Abaqus smoke test of the Python material-model bridge.
# Run this ON THE ABAQUS MACHINE (cannot run where Abaqus is absent).
#
#   ./run_abaqus_test.sh placeholder   # linear-elastic server (no JAX)
#   ./run_abaqus_test.sh nsp           # NSP-backed server (needs JAX) — *DEPVAR 12
#
# Steps: compile the C socket shim -> start the Python server -> run the
# 1-element job (umat.f) -> stop the server. Inspect one_element.dat for stress.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PROTO="$(dirname "$HERE")"
MODE="${1:-placeholder}"
PORT="${BRIDGE_PORT:-50007}"
export BRIDGE_HOST="${BRIDGE_HOST:-127.0.0.1}" BRIDGE_PORT="$PORT"

case "$MODE" in
  nsp)         SERVER="umat_server_fs_tab.py --cond DH --traj nsp_traj_DH.json"; UMAT="umat_socket_fs.f"; JOB="one_element_fs" ;;
  col)         SERVER="umat_server_col.py --profile phipg_depth_DH.json --gamma 0.4 --nramp 20"; UMAT="umat_socket_col.f"; JOB="column_dh" ;;
  col_ch)      SERVER="umat_server_col.py --profile phipg_depth_CH.json --gamma 0.4 --nramp 20"; UMAT="umat_socket_col.f"; JOB="column_ch" ;;
  patch)       SERVER="umat_server_col.py --profile phipg_flat.json --gamma 0.4 --nramp 20"; UMAT="umat_socket_col.f"; JOB="column_patch" ;;
  energy)      SERVER="umat_server_col.py --profile phipg_zero.json --gamma 0.0 --nramp 1"; UMAT="umat_socket_col.f"; JOB="energy_test" ;;
  buckle)      SERVER="umat_server_col.py --profile phipg_depth_DH.json --gamma ${BUCKLE_G:-3.0} --nramp 40"; UMAT="umat_socket_col.f"; JOB="film_buckle" ;;
  buckle_ch)   SERVER="umat_server_col.py --profile phipg_depth_CH.json --gamma ${BUCKLE_G:-3.0} --nramp 40"; UMAT="umat_socket_col.f"; JOB="film_buckle_ch" ;;
  film)        SERVER="umat_server_col.py --profile phipg_depth_DH.json --gamma 0.4 --nramp 20"; UMAT="umat_socket_col.f"; JOB="film_fs" ;;
  delam)       SERVER="umat_server_col.py --profile phipg_depth_DH.json --gamma 0.4 --nramp 40"; UMAT="umat_socket_col.f"; JOB="film_delam" ;;
  delam_ch)    SERVER="umat_server_col.py --profile phipg_depth_CH.json --gamma 0.4 --nramp 40"; UMAT="umat_socket_col.f"; JOB="film_delam_ch" ;;
  nsp_jax)     SERVER="umat_server_nsp_fs.py --cond DH"; UMAT="umat_socket_fs.f"; JOB="one_element_fs" ;;
  nsp_small)   SERVER="umat_server_nsp.py --cond DH --beta 17.0"; UMAT="umat_socket.f"; JOB="one_element" ;;
  *)           SERVER="umat_server.py"; UMAT="umat_socket.f"; JOB="one_element" ;;
esac

echo "[1/3] starting $MODE server on $BRIDGE_HOST:$PORT"
# free the port from any lingering server (a previous run's child can outlive its subshell)
pkill -f "umat_server" 2>/dev/null || true
for i in $(seq 1 10); do
  python -c "import socket;socket.socket().bind(('127.0.0.1',$PORT))" 2>/dev/null && break
  sleep 1
done
( cd "$PROTO" && exec python $SERVER --port "$PORT" ) &
SRV=$!
trap 'pkill -f "umat_server" 2>/dev/null || true; kill $SRV 2>/dev/null || true' EXIT
# wait until the server is actually listening (JAX modes need warm-up)
for i in $(seq 1 30); do
  python -c "import socket;socket.create_connection(('127.0.0.1',$PORT),0.3).close()" 2>/dev/null && { echo "  server up"; break; }
  sleep 1
done

echo "[2/3] running Abaqus job $JOB (self-contained socket UMAT $UMAT, no link step)"
cd "$HERE"
case "$MODE" in
  col|col_ch|patch) python gen_column_inp.py 8 "$JOB.inp" >/dev/null ;;
  film)             python gen_film_inp.py 12 8 "$JOB.inp" >/dev/null ;;
  delam|delam_ch)   python gen_film_delam_inp.py 12 8 "$JOB.inp" "${DELAM_TN0:-120}" "${DELAM_GC:-4.0}" >/dev/null ;;
  energy)           python gen_energy_inp.py "$JOB.inp" 0.20 >/dev/null ;;
  buckle|buckle_ch) python gen_film_buckle_inp.py 16 8 "$JOB.inp" >/dev/null ;;
esac
rm -f "$JOB".lck 2>/dev/null || true
abaqus job="$JOB" user="$UMAT" interactive ask_delete=OFF

echo "[3/3] result summary:"
if { [ "$MODE" = "buckle" ] || [ "$MODE" = "buckle_ch" ]; } && [ -f "$JOB.odb" ]; then
  abaqus python extract_buckle.py "$JOB" 2>/dev/null || echo "  (run: abaqus python extract_buckle.py $JOB)"
elif { [ "$MODE" = "delam" ] || [ "$MODE" = "delam_ch" ]; } && [ -f "$JOB.odb" ]; then
  abaqus python extract_delam.py "$JOB" 12 8 2>/dev/null || echo "  (run: abaqus python extract_delam.py $JOB 12 8)"
elif [ "$MODE" = "film" ] && [ -f "$JOB.odb" ]; then
  abaqus python extract_film.py "$JOB" 2>/dev/null || echo "  (run: abaqus python extract_film.py $JOB)"
elif { [ "$MODE" = "col" ] || [ "$MODE" = "col_ch" ] || [ "$MODE" = "patch" ]; } && [ -f "$JOB.odb" ]; then
  abaqus python extract_col.py "$JOB" 2>/dev/null || echo "  (run: abaqus python extract_col.py $JOB)"
elif [ "$JOB" = "one_element_fs" ] && [ -f one_element_fs.odb ]; then
  abaqus python extract_fs.py 2>/dev/null || echo "  (run: abaqus python extract_fs.py)"
else
  grep -A8 -iE "S11|MISES|STRESS|THE FOLLOWING" "$JOB".dat 2>/dev/null | head -40 || \
    echo "  (parse $JOB.odb for S, SDV, U manually)"
fi
echo "done."
