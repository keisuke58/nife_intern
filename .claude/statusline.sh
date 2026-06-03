#!/usr/bin/env bash
# Claude Code statusline for the NIFE project.
# Reads the harness JSON on stdin; prints: <dir> | <git branch>[±dirty] | HPC:<n> | <model>
# Kept FAST: no network calls. The HPC job count is read from a cache file that the
# /hpc command (or the hpc-dispatch agent) writes to /tmp/hpc_jobcount; stale-but-cheap.
input=$(cat)
cwd=$(printf '%s' "$input" | sed -n 's/.*"cwd"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
model=$(printf '%s' "$input" | sed -n 's/.*"display_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
[ -z "$cwd" ] && cwd="$PWD"
dir=$(basename "$cwd")

branch=""; dirty=""
if git -C "$cwd" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  branch=$(git -C "$cwd" rev-parse --abbrev-ref HEAD 2>/dev/null)
  n=$(git -C "$cwd" status --porcelain 2>/dev/null | grep -c .)
  [ "$n" -gt 0 ] 2>/dev/null && dirty=" \033[33m±${n}\033[0m"
fi

hpc=""
if [ -f /tmp/hpc_jobcount ]; then
  jc=$(cat /tmp/hpc_jobcount 2>/dev/null)
  [ -n "$jc" ] && hpc=" \033[36m| HPC:${jc}\033[0m"
fi

printf "\033[1;34m%s\033[0m" "$dir"
[ -n "$branch" ] && printf " \033[32m| %s\033[0m%b" "$branch" "$dirty"
printf "%b" "$hpc"
[ -n "$model" ] && printf " \033[90m| %s\033[0m" "$model"
