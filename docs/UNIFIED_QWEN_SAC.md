# Unified Qwen + SAC Stack

## Scope

This fork adds a shared-backbone `Qwen + SAC` training path on top of the upstream Assetto Corsa Gym codebase.

Current validated state:

- Python runtime is managed with `uv`
- main repo runtime is pinned to Python `3.10`
- real `Qwen3.5` models load in-process
- the live shared-backbone trainer runs with `Qwen/Qwen3.5-2B-Base`
- the old internal fallback encoder is no longer required for the normal path

## Layout

Key directories added in this fork:

- `planner/`
  - shared-backbone runtime
  - state tokenizer and temporal compressor
  - plan, value, and risk heads
  - segment schemas and dataset helpers
- `coach/`
  - local JSON coach API
- `dashboard/`
  - LAN-accessible training dashboard
- `qwen_runtime/`
  - isolated Python `3.10` sidecar/runtime used during bring-up and model validation

Key entrypoints:

- `train.py`
  - main multi-episode trainer
- `train_for_duration.py`
  - wall-clock-limited trainer
- `offline_build_segment_dataset.py`
  - build segment records from stored laps
- `offline_train_unified_backbone.py`
  - offline backbone/head pretraining

## Runtime Contract

- Use `uv` for all Python commands
- Main repo:
  - `.python-version` = `3.10`
  - `pyproject.toml` is the dependency source of truth
  - `uv.lock` is committed
- Preferred commands:

```powershell
uv sync --extra planner --extra coach --group dev
uv run --no-sync python train.py --algo shared_sac --config config.yml
uv run --no-sync python train_for_duration.py --algo shared_sac --config config.yml
uv run --no-sync python -m dashboard.app --outputs-root outputs --host 0.0.0.0 --port 8090
```

## Active Model Path

The intended live path is:

1. `StateTokenizer` converts numeric sim state into learned frame tokens.
2. `TemporalCompressor` produces summary tokens.
3. `BackboneWrapper` runs real Qwen up to the configured branch depth.
4. `SharedBackboneSAC` consumes:
   - `z_mid`
   - plan-code embeddings
   - value/risk outputs
   - projected current and delta observations
5. SAC actor and Q heads on GPU1 produce the actual controls.

Current practical default:

- `Qwen/Qwen3.5-2B-Base`
- `fp16`
- backbone on `cuda:0`
- SAC heads on `cuda:1`

`4B` loading works, but the current single-machine live loop has poorer latency margins. `2B` is the stable default until the hot-path latency is reduced further.

## Important Fixes In This Fork

- main repo moved from Python `3.9` to `3.10`
- `transformers` upgraded to a build that supports `qwen3_5`
- shared-backbone loader now reads `Qwen3.5` config dimensions correctly
- runtime no longer silently falls back because `hidden_size`/`num_hidden_layers` were `None`
- checkpoint bundles save only local runtime modules, not full Qwen base weights
- `train.py` now normalizes `work_dir` paths correctly
- runtime dtype handling is fixed for `fp16` Qwen with `float32` SAC-side heads
- mid-run backbone hot-swap is disabled to prevent hidden-size mismatches

## Current Known Limits

- planner latency is still high for live control:
  - roughly `370-400 ms` p95 in the current `2B` run
- the car still tends to fail on launch/reset and then overrun simple turns
- launch-assist/control handoff remains the main environment-side instability
- the hybrid path is functional, but policy quality is not yet competitive

## Dashboard

Dashboard entrypoint:

- `dashboard/app.py`

Expected usage:

```powershell
uv run --no-sync python -m dashboard.app --outputs-root outputs --host 0.0.0.0 --port 8090
```

Main tabs:

- `Overview`
- `ML`
- `Architecture`
- `Training`
- `Language`

The `ML` tab is the main debugging view for:

- losses
- alpha / entropy / Q values
- throughput
- planner latency and confidence
- run hyperparameters

## Recommended Next Work

1. Reduce shared-backbone latency before pushing longer live runs.
2. Cache or amortize planner/backbone features more aggressively for replay updates.
3. Fix launch-assist/reset reliability in the environment.
4. Only then resume long-horizon Qwen + SAC training sweeps.
