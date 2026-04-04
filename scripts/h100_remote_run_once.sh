#!/usr/bin/env bash
set -euo pipefail

workdir=${WORKDIR:-/workspace/parameter-golf}
session_name=${SESSION_NAME:-h100_run}
run_log=${RUN_LOG:-run.log}
run_id=${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}

max_wallclock_seconds=${MAX_WALLCLOCK_SECONDS:-600}
train_log_every=${TRAIN_LOG_EVERY:-200}
val_loss_every=${VAL_LOSS_EVERY:-0}
max_sw_val_tokens=${MAX_SW_VAL_TOKENS:-1000000}
sw_stride=${SW_STRIDE:-128}
sw_batch_size=${SW_BATCH_SIZE:-16}

replace_existing=${SESSION_REPLACE:-0}

cd "$workdir"

if tmux has-session -t "$session_name" 2>/dev/null; then
  if [[ "$replace_existing" == "1" ]]; then
    tmux kill-session -t "$session_name"
  else
    printf 'tmux session already exists: %s\n' "$session_name" >&2
    exit 1
  fi
fi

extra_env_names=(
  MATRIX_LR
  SCALAR_LR
  EMBED_LR
  HEAD_LR
  TIED_EMBED_LR
  MUON_WEIGHT_DECAY
  MUON_MOMENTUM
  QK_GAIN_INIT
  NUM_LAYERS
  MODEL_DIM
  NUM_HEADS
  NUM_KV_HEADS
  MLP_MULT
  MLP_ACT
  TRAIN_BATCH_TOKENS
  TRAIN_SEQ_LEN
  WARMUP_STEPS
  WARMDOWN_ITERS
  SEED
)
extra_env_cmd=""
for name in "${extra_env_names[@]}"; do
  if [[ -n "${!name:-}" ]]; then
    printf -v extra_env_cmd '%s%s=%q \\\n' "$extra_env_cmd" "$name" "${!name}"
  fi
done

cmd=$(
  cat <<EOF
cd '$workdir' && rm -f '$run_log' final_model.pt final_model.int8.ptz && \
RUN_ID='$run_id' \
MAX_WALLCLOCK_SECONDS='$max_wallclock_seconds' \
TRAIN_LOG_EVERY='$train_log_every' \
VAL_LOSS_EVERY='$val_loss_every' \
MAX_SW_VAL_TOKENS='$max_sw_val_tokens' \
SW_STRIDE='$sw_stride' \
SW_BATCH_SIZE='$sw_batch_size' \
$extra_env_cmd\
python3 -u train_gpt.py 2>&1 | tee '$run_log'
EOF
)

tmux new-session -d -s "$session_name" "$cmd"

printf 'started session=%s run_id=%s log=%s/%s\n' \
  "$session_name" "$run_id" "$workdir" "$run_log"
printf 'eval_mode=max_sw_val_tokens=%s sw_stride=%s sw_batch_size=%s\n' \
  "$max_sw_val_tokens" "$sw_stride" "$sw_batch_size"
