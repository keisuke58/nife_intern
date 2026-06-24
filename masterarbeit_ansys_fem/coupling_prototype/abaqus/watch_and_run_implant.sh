#!/usr/bin/env bash
# QXT ライセンス待機 → 空き次第インプラントUMATジョブ自動実行
# Usage: bash watch_and_run_implant.sh [DH|CH] [thread|flat]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PROTO="$HERE/.."
ABAQUS="/home/nishioka/DassaultSystemes/SIMULIA/Commands/abaqus"
LOG="$HERE/watch_implant.log"

COND="${1:-DH}"
PROF="${2:-thread}"
PORT=50007
HF=1.0
INP="implant_umat_${COND}_${PROF}"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== watch_and_run_implant.sh started: cond=$COND profile=$PROF ==="

# --- 1. QXT が空くまで待つ ---
while true; do
    QXT_FREE=$($ABAQUS licensing dslsstat -usage 2>/dev/null \
        | grep '| QXT' | awk -F'|' '{gsub(/[[:space:]]/,"",$7); print 50-int($7)}' || echo 0)
    log "QXT 空きトークン: $QXT_FREE / 50"
    if [ "$QXT_FREE" -ge 2 ]; then
        log "QXT 利用可能 → ジョブ開始"
        break
    fi
    sleep 300   # 5分ごとに再チェック
done

cd "$HERE"

# --- 2. inp 生成 ---
log "inp 生成中 ..."
python "$PROTO/gen_implant_umat_inp.py" "$COND" "$PROF" "${INP}.inp" 2>&1 | tee -a "$LOG"

# --- 3. Python UMAT サーバ起動 ---
log "Python UMAT サーバ起動 (port=$PORT, coord-axis=1) ..."
python "$PROTO/umat_server_col.py" \
    --port $PORT \
    --coord-axis 1 \
    --height $HF \
    --profile "$PROTO/phipg_depth_${COND}.json" \
    --gamma 0.4 --nramp 20 --growth_mode iso >> "$LOG" 2>&1 &
SERVER_PID=$!
log "server PID=$SERVER_PID"
sleep 3

# --- 4. Abaqus ジョブ投入 ---
log "Abaqus 投入 (detached, NLGEOM=YES, C3D8 $COND/$PROF) ..."
$ABAQUS job="$INP" input="${INP}.inp" \
    user=umat_socket_col.f cpus=1 double=both \
    scratch="$HERE" ask_delete=OFF background >> "$LOG" 2>&1 || true

# --- 5. .sta ポーリング（最大 60 分）---
log ".sta ポーリング開始 ..."
DONE=0
for i in $(seq 1 720); do
    sleep 5
    if [ -f "${INP}.sta" ]; then
        LAST=$(tail -1 "${INP}.sta")
        log "  sta[$i]: $LAST"
        if echo "$LAST" | grep -q "THE ANALYSIS HAS COMPLETED"; then
            log "✅ 解析完了"
            DONE=1; break
        fi
        if echo "$LAST" | grep -q "HAS NOT BEEN COMPLETED"; then
            log "❌ 解析失敗 — ${INP}.msg を確認"
            break
        fi
    fi
done

# --- 6. サーバ停止 ---
log "サーバ停止 (PID=$SERVER_PID) ..."
kill $SERVER_PID 2>/dev/null || true

# --- 7. 結果抽出 & プロット ---
if [ "$DONE" -eq 1 ]; then
    log "結果抽出 & プロット ..."
    python "$HERE/extract_implant_umat.py" "$INP" >> "$LOG" 2>&1 && \
        log "✅ 図保存: ${INP}_stress.png" || \
        log "⚠️  extract 失敗 — odb を手動確認"
fi

log "=== 完了: $INP  ==="
log "主要ファイル: ${INP}.odb  ${INP}_stress.png  ${INP}.sta"
