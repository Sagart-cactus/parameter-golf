#!/usr/bin/env bash
set -euo pipefail

workdir=${WORKDIR:-/workspace/parameter-golf}
results_file=${RESULTS_FILE:-results_h100.tsv}

cd "$workdir"

mkdir -p logs

touch .git/info/exclude
for pattern in \
  "$results_file" \
  "run.log" \
  "final_model.pt" \
  "final_model.int8.ptz"
do
  if ! grep -Fxq "$pattern" .git/info/exclude; then
    printf '%s\n' "$pattern" >> .git/info/exclude
  fi
done

if [[ ! -f "$results_file" ]]; then
  printf 'commit\tval_bpb\tsw_bpb\tcompressed_mb\tstatus\tdescription\n' > "$results_file"
fi

python3 --version
git status --short --branch
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
find data/datasets/fineweb10B_sp1024 -maxdepth 1 -name 'fineweb_train_*.bin' | wc -l | awk '{print "train_shards=" $1}'
find data/datasets/fineweb10B_sp1024 -maxdepth 1 -name 'fineweb_val_*.bin' | wc -l | awk '{print "val_shards=" $1}'
printf 'results_file=%s\n' "$results_file"
