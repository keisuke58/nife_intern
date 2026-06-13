#!/usr/bin/env bash
# Rigorous implant-thread FEM sweep: mesh convergence + thread-amplitude (curvature) sweep + main fields.
# Smooth sinusoidal thread, isotropic volume-matched growth, base-bonded patch (free sides).
set -e
export PATH="$HOME/DassaultSystemes/SIMULIA/Commands:$PATH"
cd "$(dirname "$0")"
PER=3

run() { # cond profile job Nx Nz amp
  python gen_implant_inp.py "$1" "$2" "$3.inp" "$4" "$5" "$PER" "$6" iso >/dev/null
  timeout 400 abaqus job="$3" interactive ask_delete=OFF >/dev/null 2>&1
  abaqus python extract_implant.py "$3" "$4" 2>&1 | grep "interface peak"
}

echo "=== MESH CONVERGENCE (DH thread, amp=0.18) ==="
run DH thread imp_conv_n36  36  6  0.18
run DH thread imp_conv_n72  72 12  0.18
run DH thread imp_conv_n108 108 18 0.18
run DH thread imp_conv_n144 144 24 0.18

echo "=== AMPLITUDE / CURVATURE SWEEP (DH, n108) ==="
run DH thread imp_amp_000 108 18 0.0
run DH thread imp_amp_060 108 18 0.06
run DH thread imp_amp_120 108 18 0.12
run DH thread imp_amp_300 108 18 0.30
run DH thread imp_amp_450 108 18 0.45

echo "=== MAIN FIELDS (n108, amp=0.18) ==="
run DH thread implant_DH_thread 108 18 0.18
run CH thread implant_CH_thread 108 18 0.18
abaqus python extract_implant_field.py implant_DH_thread 2>&1 | grep elements
abaqus python extract_implant_field.py implant_CH_thread 2>&1 | grep elements
echo "DONE"
