#!/usr/bin/env bash
# Drives the two-way mechanotransduction Abaqus column on a single composition / mesh / load case.
# Spawns umat_server_twoway.py, runs a column deck with umat_socket_twoway.f, then extracts the
# substratum stress AND the final φ_Pg(z) profile so that the redistribution (Pg sequesters away from
# the high-σ substratum into the lower-σ bulk) can be quantified against the one-way headline.
#
#   cd /home/nishioka/IKM_Hiwi/FEM/coupling_prototype/abaqus
#   bash run_twoway.sh DH 0.5    # cond=DH, σ_c=0.5
set -euo pipefail
cd "$(dirname "$0")"

ABQ=~/DassaultSystemes/SIMULIA/Commands/abaqus
COND=${1:-DH}
SIGMA_C=${2:-0.5}
PORT=50008
NAME="twoway_col_${COND}_sc${SIGMA_C//./p}"

# generate the (existing) column input deck, plug in the two-way socket UMAT
GSCALE=0.05 HEIGHT=4.0 python gen_column_phi_inp.py "$NAME" 16 2>&1 | tail -2
# the deck targets umat_growth_phi.f by default; swap the *USER MATERIAL block so it routes through
# the socket UMAT instead (sed-replace the user material include line)
sed -i "s|umat_growth_phi|umat_socket_twoway|g" "${NAME}.inp"

# start the python server in the background; wait for ready banner
python ../umat_server_twoway.py --port "$PORT" --cond "$COND" \
       --beta 0.05 --sigma-c "$SIGMA_C" >server_twoway.log 2>&1 &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; true" EXIT
sleep 2

echo "=== solve $NAME with two-way socket UMAT (cond=$COND, σ_c=$SIGMA_C) ==="
rm -f "${NAME}.lck"
"$ABQ" job="$NAME" user=umat_socket_twoway.f cpus=1 interactive \
    >"${NAME}.log" 2>&1 || true

if grep -q COMPLETED "${NAME}.log"; then
    "$ABQ" python extract_plateau.py "$NAME" >"${NAME}_plateau.out" 2>&1 || true
    echo "--- substratum plateau ---"
    cat "${NAME}_plateau.out"
else
    echo "$NAME FAILED"
    tail -30 "${NAME}.dat" 2>/dev/null || true
fi

# stop the server cleanly
kill $SERVER_PID 2>/dev/null || true
echo "=== two-way run done ==="
