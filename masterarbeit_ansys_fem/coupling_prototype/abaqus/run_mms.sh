#!/usr/bin/env bash
# A3 MMS order-of-accuracy study: one growth-off server, sweep Nx, fit log(L2err) vs log(h).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; PROTO="$(dirname "$HERE")"
PORT="${BRIDGE_PORT:-50007}"
export BRIDGE_HOST=127.0.0.1 BRIDGE_PORT="$PORT"
UMAT="umat_socket_col.f"

pkill -f "umat_server" 2>/dev/null || true
for i in $(seq 1 10); do python -c "import socket;socket.socket().bind(('127.0.0.1',$PORT))" 2>/dev/null && break; sleep 1; done
( cd "$PROTO" && exec python umat_server_col.py --profile phipg_zero.json --gamma 0.0 --nramp 1 --port "$PORT" ) &
SRV=$!
trap 'pkill -f "umat_server" 2>/dev/null || true; kill $SRV 2>/dev/null || true' EXIT
for i in $(seq 1 30); do python -c "import socket;socket.create_connection(('127.0.0.1',$PORT),0.3).close()" 2>/dev/null && break; sleep 1; done

cd "$HERE"
OUT="mms_convergence.txt"; : > "$OUT"
for N in 4 8 16 32; do
  JOB="mms_N$N"
  python gen_mms_inp.py "$N" "$JOB.inp" >/dev/null
  rm -f "$JOB".lck 2>/dev/null || true
  abaqus job="$JOB" user="$UMAT" interactive ask_delete=OFF >/dev/null 2>&1
  abaqus python extract_mms.py "$JOB" "$N" >> "$OUT"
  echo "  Nx=$N done"
done
echo "== Nx  h  L2err =="; cat "$OUT"
python - "$OUT" <<'PY'
import sys, math
rows=[l.split() for l in open(sys.argv[1]) if l.strip()]
h=[float(r[1]) for r in rows]; e=[float(r[2]) for r in rows]
print("observed orders p (successive pairs):")
for i in range(1,len(h)):
    p=math.log(e[i-1]/e[i])/math.log(h[i-1]/h[i])
    print("  Nx %s->%s :  p = %.2f" % (rows[i-1][0], rows[i][0], p))
# least-squares slope over all points
n=len(h); lx=[math.log(v) for v in h]; ly=[math.log(v) for v in e]
mx=sum(lx)/n; my=sum(ly)/n
slope=sum((lx[i]-mx)*(ly[i]-my) for i in range(n))/sum((lx[i]-mx)**2 for i in range(n))
print("least-squares order over all meshes:  p = %.2f  (expect ~2 for C3D8 L2)" % slope)
PY
