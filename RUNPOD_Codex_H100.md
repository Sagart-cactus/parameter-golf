# RunPod H100 Codex Loop

This workflow keeps Codex in charge of model changes from the local machine while the pod only does durable execution.

## Principle

- Codex edits and judges locally.
- The pod only runs one experiment at a time in `tmux`.
- Each remote run survives SSH disconnects.
- Cheap eval settings are used for the inner loop.
- Full `5M` / `stride=64` sliding-window eval is reserved for promoted baseline runs.

## One-Time Pod Bootstrap

On the pod:

```bash
cd /workspace/parameter-golf
scripts/h100_remote_bootstrap.sh
```

This ensures:

- `results_h100.tsv` exists
- `run.log`, `final_model.pt`, and `final_model.int8.ptz` stay untracked
- GPU/runtime state is printed

## Codex-Supervised Loop

1. Codex edits `train_gpt.py` locally.
2. Codex commits and pushes the branch to GitHub.
3. On the pod, pull the branch:

```bash
git fetch origin
git checkout autoresearch/h100-2026-04-04-baseline
git pull --ff-only origin autoresearch/h100-2026-04-04-baseline
```

4. Start one remote run in `tmux`:

```bash
SESSION_NAME=h100_loop \
MAX_WALLCLOCK_SECONDS=600 \
MAX_SW_VAL_TOKENS=1000000 \
SW_STRIDE=128 \
SW_BATCH_SIZE=16 \
scripts/h100_remote_run_once.sh
```

5. Check status:

```bash
SESSION_NAME=h100_loop scripts/h100_remote_status.sh
```

6. Parse the completed log and append a row:

```bash
python3 scripts/parse_h100_log.py run.log --summary
python3 scripts/parse_h100_log.py run.log \
  --tsv-row \
  --commit "$(git rev-parse --short HEAD)" \
  --status keep \
  --description "manual note" >> results_h100.tsv
```

## Eval Presets

Use these presets on `1x H100`.

### Loop Mode

For continuous experimentation:

```bash
MAX_SW_VAL_TOKENS=1000000
SW_STRIDE=128
SW_BATCH_SIZE=16
```

This keeps total wall-clock much lower while preserving ranking signal.

### Reference Mode

For promoted runs only:

```bash
MAX_SW_VAL_TOKENS=5000000
SW_STRIDE=64
SW_BATCH_SIZE=16
```

This gives a stronger baseline number, but on the current script it adds about 11 minutes of sliding-window eval after training.

## Important Constraint

True Codex-driven judgment requires an active Codex session. If the local Codex session ends, the pod can keep running the current experiment, but it cannot autonomously choose the next architectural change unless you replace that logic with a heuristic search loop.
