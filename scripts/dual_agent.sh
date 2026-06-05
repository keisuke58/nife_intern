#!/usr/bin/env bash
# dual_agent.sh — Claude Code + Codex を tmux で並列起動
# Usage: bash scripts/dual_agent.sh [nife|thesis]
set -e

SESSION="dual_agent"
NIFE="/home/nishioka/IKM_Hiwi/nife"
THESIS="/home/nishioka/LUHsummer26/30_Masterarbeit"
TARGET="${1:-nife}"

# 既存セッションがあれば attach
tmux has-session -t "$SESSION" 2>/dev/null && tmux attach -t "$SESSION" && exit 0

# 新規セッション作成
tmux new-session -d -s "$SESSION" -x 220 -y 50

# ── 上段: Claude Code ────────────────────────────────────────────────────────
tmux rename-window -t "$SESSION:0" "agents"

if [ "$TARGET" = "thesis" ]; then
  tmux send-keys -t "$SESSION:0" "cd $THESIS && claude" Enter
else
  tmux send-keys -t "$SESSION:0" "cd $NIFE && claude" Enter
fi

# ── 下段左: Codex ────────────────────────────────────────────────────────────
tmux split-window -v -t "$SESSION:0" -p 40
if [ "$TARGET" = "thesis" ]; then
  tmux send-keys -t "$SESSION:0.1" "cd $THESIS && codex" Enter
else
  tmux send-keys -t "$SESSION:0.1" "cd $NIFE && codex" Enter
fi

# ── 下段右: shell (git/HPC dispatch) ─────────────────────────────────────────
tmux split-window -h -t "$SESSION:0.1" -p 40
tmux send-keys -t "$SESSION:0.2" "cd $NIFE && echo 'shell ready'" Enter

# レイアウト調整
tmux select-pane -t "$SESSION:0.0"
tmux attach -t "$SESSION"
