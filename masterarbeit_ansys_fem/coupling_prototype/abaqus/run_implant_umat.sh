#!/usr/bin/env bash
# Run NSP UMAT on implant-thread geometry (C3D8 thin slice, pseudo-plane-strain)
# Usage: bash run_implant_umat.sh [DH|CH] [thread|flat]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PROTO="$HERE/.."
ABAQUS="${ABAQUS:-abaqus}"

COND="${1:-DH}"
PROF="${2:-thread}"
PORT=50007
HF=1.0
INP="implant_umat_${COND}_${PROF}"

cd "$HERE"

echo "[1] generating inp (C3D8 thin slice) ..."
python "$PROTO/gen_implant_umat_inp.py" "$COND" "$PROF" "${INP}.inp"

echo "[2] starting Python UMAT server (coord-axis=1 = y-depth for implant) ..."
python "$PROTO/umat_server_col.py" \
    --port $PORT \
    --coord-axis 1 \
    --height $HF \
    --profile "$PROTO/phipg_depth_${COND}.json" \
    --gamma 0.4 --nramp 20 --growth_mode iso &
SERVER_PID=$!
sleep 2

echo "[3] running Abaqus 2024 (non-interactive, detached) ..."
$ABAQUS job="$INP" input="${INP}.inp" \
    user=umat_socket_col.f cpus=1 double=both \
    scratch="$HERE" background 2>&1 | tee "${INP}_submit.log" || true

# poll .sta until complete
echo "[4] waiting for job to finish (.sta polling) ..."
for i in $(seq 1 120); do
    sleep 5
    if [ -f "${INP}.sta" ]; then
        tail -1 "${INP}.sta"
        if grep -q "THE ANALYSIS HAS COMPLETED" "${INP}.sta" 2>/dev/null; then
            echo "  -> completed"
            break
        fi
        if grep -q "THE ANALYSIS HAS NOT BEEN COMPLETED" "${INP}.sta" 2>/dev/null; then
            echo "  -> FAILED"
            break
        fi
    fi
done

echo "[5] stopping server ..."
kill $SERVER_PID 2>/dev/null || true

echo "[6] extracting results ..."
python "$HERE/extract_implant_field.py" "$INP" 2>/dev/null || \
python "$HERE/extract_col.py" "$INP" 2>/dev/null || \
echo "  (extract script not matched — check ${INP}.odb manually)"

echo "done. Key files: ${INP}.odb  ${INP}.sta  ${INP}.msg"
