#!/usr/bin/env bash
set -euo pipefail

workdir=${WORKDIR:-/workspace/parameter-golf}
session_name=${SESSION_NAME:-h100_run}
run_log=${RUN_LOG:-run.log}
tail_lines=${TAIL_LINES:-40}

cd "$workdir"

printf 'session_name=%s\n' "$session_name"
if tmux has-session -t "$session_name" 2>/dev/null; then
  printf 'tmux_session=present\n'
else
  printf 'tmux_session=absent\n'
fi

if pgrep -af 'python3 -u train_gpt.py' >/dev/null 2>&1; then
  printf 'trainer_process=running\n'
  pgrep -af 'python3 -u train_gpt.py'
else
  printf 'trainer_process=stopped\n'
fi

if [[ -f "$run_log" ]]; then
  printf 'run_log=%s/%s\n' "$workdir" "$run_log"
  tail -n "$tail_lines" "$run_log"
  if command -v python3 >/dev/null 2>&1; then
    python3 scripts/parse_h100_log.py "$run_log" --summary || true
  fi
else
  printf 'run_log_missing=%s/%s\n' "$workdir" "$run_log"
fi
