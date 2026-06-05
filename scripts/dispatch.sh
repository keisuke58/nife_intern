#!/usr/bin/env bash
# dispatch.sh — タスクを Claude Code / Codex に振り分けるヘルパー
# source this file or add to ~/.bashrc:
#   source /home/nishioka/IKM_Hiwi/nife/scripts/dispatch.sh
#
# Usage:
#   to_codex "このスクリプトをリファクタして"         # Codex に送る（新ウィンドウ）
#   to_claude "ch5 の Abstract を短くして"            # Claude Code (メモ表示)
#   both "LOO スクリプトのバグを探して"               # 両方に同じタスク

NIFE="/home/nishioka/IKM_Hiwi/nife"

# Codex にタスクを送る（バックグラウンドで新ターミナルを開く）
to_codex() {
  local task="$*"
  if tmux has-session -t dual_agent 2>/dev/null; then
    # dual_agent セッションの Codex ペインに送る
    tmux send-keys -t "dual_agent:0.1" "$task" Enter
    echo "[→ Codex] $task"
  else
    echo "[codex] セッションなし。'bash $NIFE/scripts/dual_agent.sh' で起動してください"
    echo "  または: codex \"$task\""
  fi
}

# Claude Code 用メモ（このセッションへのリマインダー）
to_claude() {
  local task="$*"
  echo ""
  echo "╔══════════════════════════════════════╗"
  echo "║  Claude Code タスク                  ║"
  echo "╟──────────────────────────────────────╢"
  echo "║  $task"
  echo "╚══════════════════════════════════════╝"
  echo ""
}

# 両方に送る
both() {
  local task="$*"
  to_codex "$task"
  to_claude "$task"
}

# タスク種別の判定ヘルパー
route() {
  local task="$*"
  # コード系キーワード → Codex
  if echo "$task" | grep -qiE 'リファクタ|デバッグ|refactor|debug|スクリプト作成|script|関数|function|エラー修正|fix bug'; then
    echo "→ Codex 推奨"
    to_codex "$task"
  # 解析・修論系 → Claude Code
  elif echo "$task" | grep -qiE '修論|thesis|解析|analysis|図|figure|LaTeX|章|chapter|スライド|slide'; then
    echo "→ Claude Code 推奨"
    to_claude "$task"
  else
    echo "判断できません。手動で to_codex または to_claude を使ってください"
    echo "  task: $task"
  fi
}

echo "[dispatch] loaded. commands: to_codex, to_claude, both, route"
