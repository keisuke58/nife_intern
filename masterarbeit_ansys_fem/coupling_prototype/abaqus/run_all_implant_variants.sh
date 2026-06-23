#!/usr/bin/env bash
# Run all NSP UMAT implant variants sequentially (one QXT token held at a time).
# Waits for QXT to be free before starting each variant.
# Usage: bash run_all_implant_variants.sh [DH|CH]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PROTO="$HERE/.."
ABAQUS="${ABAQUS:-/home/nishioka/DassaultSystemes/SIMULIA/Commands/abaqus}"
LOG="$HERE/run_all_variants.log"
PORT=50007
COND="${1:-DH}"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

qxt_free() {
    $ABAQUS licensing dslsstat -usage 2>/dev/null \
        | grep '| QXT' | awk -F'|' '{gsub(/[[:space:]]/,"",$7); print 50-int($7)}' || echo 0
}

wait_qxt() {
    while true; do
        F=$(qxt_free)
        log "QXT 空き: $F / 50"
        [ "$F" -ge 2 ] && break
        sleep 300
    done
}

run_variant() {
    local INP="$1"; local GEN="$2"; local GEN_ARGS="$3"; local SERVER_ARGS="$4"
    local UMAT="${5:-umat_socket_col.f}"
    log "=== $INP ==="
    wait_qxt
    log "inp生成: $GEN $GEN_ARGS"
    python "$PROTO/$GEN" $GEN_ARGS "${INP}.inp" 2>&1 | tee -a "$LOG"
    log "サーバー起動 (port=$PORT) ..."
    python "$PROTO/umat_server_col.py" --port $PORT $SERVER_ARGS >> "$LOG" 2>&1 &
    SRV=$!
    sleep 3
    log "Abaqus投入: $INP (umat=$UMAT)"
    $ABAQUS job="$INP" input="${INP}.inp" user="$UMAT" \
        cpus=1 double=both scratch="$HERE" ask_delete=OFF background >> "$LOG" 2>&1 || true
    log ".sta ポーリング ..."
    DONE=0
    for i in $(seq 1 720); do
        sleep 5
        [ -f "${INP}.sta" ] || continue
        LAST=$(tail -1 "${INP}.sta")
        log "  sta[$i]: $LAST"
        echo "$LAST" | grep -q "THE ANALYSIS HAS COMPLETED"     && { DONE=1; break; }
        echo "$LAST" | grep -q "HAS NOT BEEN COMPLETED"         && { log "❌ 失敗"; break; }
    done
    kill $SRV 2>/dev/null || true
    if [ "$DONE" -eq 1 ]; then
        python "$HERE/extract_implant_umat.py" "$INP" >> "$LOG" 2>&1 \
            && log "✅ $INP 完了 → ${INP}_stress.png" \
            || log "⚠️  extract失敗"
    fi
}

cd "$HERE"
log "=== run_all_implant_variants.sh start: cond=$COND ==="

# ---- 1. 基本 thread / flat (DH と CH) ----
run_variant "implant_umat_${COND}_thread" "gen_implant_umat_inp.py" "$COND thread" \
    "--coord-axis 1 --height 1.0 --profile ${PROTO}/phipg_depth_${COND}.json --gamma 0.4 --nramp 20 --growth_mode iso"

run_variant "implant_umat_${COND}_flat" "gen_implant_umat_inp.py" "$COND flat" \
    "--coord-axis 1 --height 1.0 --profile ${PROTO}/phipg_depth_${COND}.json --gamma 0.4 --nramp 20 --growth_mode iso"

# ---- 2. RVE (周期 BC) ----
run_variant "implant_umat_rve_${COND}_thread" "gen_implant_umat_rve_inp.py" "$COND thread" \
    "--coord-axis 1 --height 1.0 --profile ${PROTO}/phipg_depth_${COND}.json --gamma 0.4 --nramp 20 --growth_mode iso"

# ---- 3. 軸対称 (CGAX4, R0=2.0) ----
run_variant "implant_umat_axi_${COND}_R2.0" "gen_implant_umat_axi_inp.py" "$COND 2.0" \
    "--coord-axis 0 --coord-offset 2.0 --height 1.0 --profile ${PROTO}/phipg_depth_${COND}.json --gamma 0.4 --nramp 20"

# ---- 4. 粘弾性 (VE, umat_socket_ve.f) ----
run_variant "implant_umat_ve_${COND}_thread" "gen_implant_umat_ve_inp.py" "$COND thread" \
    "--coord-axis 1 --height 1.0 --profile ${PROTO}/phipg_depth_${COND}.json --gamma 0.4 --nramp 20 --prony ${PROTO}/prony_biofilm.json" \
    "umat_socket_ve.f"

# ---- 5. 3D 螺旋 (小メッシュ版) ----
run_variant "implant_umat_3d_${COND}" "gen_implant_umat_3d_inp.py" "$COND" \
    "--geometry helical --R0 2.0 --pitch 1.0 --amp 0.18 --height 1.0 --profile ${PROTO}/phipg_depth_${COND}.json --gamma 0.4 --nramp 20"

log "=== 全バリアント完了: cond=$COND ==="
