# Unified Qwen + SAC Stack

## Scope

This fork adds a shared-backbone `Qwen + SAC` training path on top of the upstream Assetto Corsa Gym codebase.

Current validated state (March 14, 2026):

- Python runtime managed with `uv`, pinned to Python `3.10`
- Real `Qwen 3.5` models load in-process (2B and 4B verified)
- Live shared-backbone trainer runs with `Qwen/Qwen3.5-2B-Base`
- The old internal fallback encoder is no longer required for the normal path
- Multiple live training runs completed with real Qwen backbone
- Dashboard operational on port 8090

## Layout

Key directories added in this fork:

- `planner/`
  - Shared-backbone runtime (`unified_backbone.py`)
  - StateTokenizer: converts numeric sim state into learned frame tokens
  - TemporalCompressor: GRU + attention pooling over frame buffer
  - BackboneWrapper: runs real Qwen up to configured branch depth
  - PlanHead (6 plan-code fields), ValueHead, RiskHead
  - Segment schemas and dataset helpers
  - Hindsight labeler for ground-truth plan codes from episode outcomes
- `coach/`
  - Local JSON coach API for context-aware coaching during training
- `dashboard/`
  - LAN-accessible FastAPI training dashboard
  - Live metrics, losses, entropy, Q values, throughput, planner latency
- `qwen_runtime/`
  - Isolated Python 3.10 sidecar used during bring-up and model validation

Key entrypoints:

- `train.py` - main multi-episode trainer (`--algo shared_sac`)
- `train_for_duration.py` - wall-clock-limited trainer
- `offline_build_segment_dataset.py` - build segment records from stored laps
- `offline_train_unified_backbone.py` - offline backbone/head pretraining

## Runtime Contract

- Use `uv` for all Python commands
- `.python-version` = `3.10`
- `pyproject.toml` is the dependency source of truth
- `uv.lock` is committed

Preferred commands:

```powershell
uv sync --extra planner --extra coach --group dev
uv run --no-sync python train.py --algo shared_sac --config config.yml
uv run --no-sync python train_for_duration.py --algo shared_sac --config config.yml
uv run --no-sync python -m dashboard.app --outputs-root outputs --host 0.0.0.0 --port 8090
```

## Active Model Path

The intended live path is:

1. `StateTokenizer` converts numeric sim state into learned frame tokens
2. `TemporalCompressor` produces summary tokens from frame buffer (64 frames)
3. `BackboneWrapper` runs real Qwen up to configured branch depth
4. `SharedBackboneSAC` consumes:
   - `z_mid` (512-dim) from backbone branch layer
   - Plan-code embeddings from PlanHead
   - Value/risk outputs from auxiliary heads
   - Projected current and delta observations
5. FiLM-conditioned SAC actor and Q heads produce the actual controls

Current practical default:

- `Qwen/Qwen3.5-2B-Base`
- `fp16` with 4-bit NF4 quantization
- Backbone on `cuda:0`
- SAC heads on `cuda:1`
- LoRA adapters: rank 16, alpha 32

`4B` loading works, but the current single-machine live loop has poorer latency margins. `2B` is the stable default until hot-path latency is reduced further.

## Key Configuration (`config.yml`)

```yaml
UnifiedBackbone:
  enabled: True
  model_name: "Qwen/Qwen3.5-4B-Base"
  fallback_model_name: "Qwen/Qwen3.5-2B-Base"
  quantization: "4bit_nf4"
  backbone_device_index: 0
  head_device_index: 1
  frame_buffer_len: 64
  summary_token_count: 8
  cache_refresh_hz: 5.0
  branch_layer_4b: 16
  branch_layer_2b: 12
  lora_enabled: True
  lora_rank: 16
  lora_alpha: 32
  lora_lr: 1e-5
  head_lr: 3e-4
  backbone_update_every: 100
  student_distill_weight: 0.5
  enable_grad_checkpointing: True
```

## Important Fixes In This Fork

- Main repo moved from Python `3.9` to `3.10`
- `transformers` upgraded to a build that supports `qwen3_5`
- Shared-backbone loader reads `Qwen3.5` config dimensions correctly
- Runtime no longer silently falls back because `hidden_size`/`num_hidden_layers` were `None`
- Checkpoint bundles save only local runtime modules, not full Qwen base weights
- `train.py` normalizes `work_dir` paths correctly
- Runtime dtype handling fixed for `fp16` Qwen with `float32` SAC-side heads
- Mid-run backbone hot-swap disabled to prevent hidden-size mismatches

## Current Known Limits

- Planner latency still high for live control (~370-400ms p95 with 2B)
- Car fails simple turns and does not complete clean laps
- Launch-assist / control handoff remains main environment-side instability
- Policy quality not yet competitive with pure SAC baselines on reward
- No run has completed a clean timed lap

## Dashboard

Dashboard entrypoint:

- `dashboard/app.py`

Usage:

```powershell
uv run --no-sync python -m dashboard.app --outputs-root outputs --host 0.0.0.0 --port 8090
```

Main tabs:

- Overview - process status
- ML - losses, alpha, entropy, Q values, throughput
- Architecture - model info, layer details
- Training - hyperparameters, run config
- Language - plan codes, backbone outputs

## Recommended Next Work

1. Reduce shared-backbone latency (cache/amortize planner features for replay updates)
2. Stabilize launch/reset behavior in the environment
3. Change throttle/brake from relative to absolute control
4. Add curriculum with hard-corner entry resets
5. Resume longer real-Qwen training sweeps only after latency and reset improvements
