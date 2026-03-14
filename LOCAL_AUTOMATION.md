# Local Automation Notes

This file documents the local changes inside the nested gym checkout at `C:\Workspace\RacingSim\assetto_corsa_gym`.

The canonical end-to-end Windows setup guide lives at:
- [../docs/SETUP_AND_TRAINING.md](../docs/SETUP_AND_TRAINING.md)

Current project status and training history:
- [../docs/STATUS.md](../docs/STATUS.md)

## Runtime

This repo uses `uv` for Python environment management:

- `.python-version` = `3.10`
- `pyproject.toml` is the dependency source of truth
- `uv.lock` is committed for reproducible installs

Install:

```powershell
uv sync --extra planner --extra coach --group dev
```

## Added Scripts

- [smoke_random.py](smoke_random.py) - Live wiring smoke test against the running sim
- [train.py](train.py) - Main multi-episode trainer (supports `--algo shared_sac`)
- [train_for_duration.py](train_for_duration.py) - Wall-clock-limited trainer with checkpoint resume and replay buffer seeding
- [offline_pretrain.py](offline_pretrain.py) - Offline SAC pretrain against the public HuggingFace dataset
- [offline_train_unified_backbone.py](offline_train_unified_backbone.py) - Offline Qwen + SAC backbone/head pretraining
- [offline_build_segment_dataset.py](offline_build_segment_dataset.py) - Build segment records from stored laps
- [plot_live_run.py](plot_live_run.py) - Exports metrics plots, scalars CSV, and summary JSON for a finished run
- [live_experiment_sweep.py](live_experiment_sweep.py) - Multi-config experiment sweep automation
- [curate_top_laps.py](curate_top_laps.py) - Top lap curation utility
- [live_train_watch.ps1](live_train_watch.ps1) - Waits for plugin TCP port, then runs smoke + short SAC job

## Added Directories

- `planner/` - Unified Qwen backbone runtime
  - `unified_backbone.py` - StateTokenizer, TemporalCompressor, BackboneWrapper, plan/value/risk heads
  - `schemas.py` - Data structures for segments, plan codes, backbone outputs
  - `segment_dataset.py` - Episode segmentation for hindsight labeling
  - `hindsight_labeler.py` - Ground-truth plan code generation from episode outcomes
- `coach/` - Local JSON coach API for context-aware coaching during training
- `dashboard/` - FastAPI web dashboard for live training monitoring (port 8090)
- `qwen_runtime/` - Isolated Python 3.10 sidecar for Qwen model validation and inference

## Local Code Changes

- [ac_env.py](assetto_corsa_gym/AssettoCorsaEnv/ac_env.py)
  - Added post-reset launch assist: holds throttle through vJoy until the automatic gearbox engages
  - Action prior sampling, state-action bias, curvature-aware shaping
  - Unified backbone guidance integration
  - Overspeed / turn-gap / heading-error penalties
  - Reference steering bias
  - Teacher controller blending
- [sensors_par.py](assetto_corsa_gym/AssettoCorsaPlugin/plugins/sensors_par/sensors_par.py)
  - Added guards for missing CSP helper APIs
- [WASD.ini](assetto_corsa_gym/AssettoCorsaPlugin/windows-libs/WASD.ini)
  - Removed stale vJoy bindings from the keyboard profile
- [agent.py](algorithm/discor/discor/agent.py)
  - SharedBackboneSAC integration, post-episode update budget
- [shared_backbone_sac.py](algorithm/discor/discor/algorithm/shared_backbone_sac.py)
  - FiLM-conditioned SAC policy and Q-networks consuming backbone z_mid
  - Plan-code embedding, confidence weighting
- [replay_buffer.py](algorithm/discor/discor/replay_buffer.py)
  - Extended for unified backbone (previous-state storage, segment metadata)
- [config.yml](config.yml)
  - Full UnifiedBackbone, Coach, Knowledge configuration blocks
  - Action prior, reward shaping, teacher controller parameters

## Outer Launcher Script

The stock-launcher automation lives outside this checkout at:

- [start_acgym_supported.ps1](../start_acgym_supported.ps1)
- [scripts/](../scripts/)

That script:

- Detects the real AC config root under Documents / OneDrive
- Forces `monza` + `ks_mazda_miata` Hotlap
- Enables `sensors_par`
- Copies the `Vjoy.ini` preset into the active `controls.ini`
- Rewrites `PGUID0` to the currently enumerated live `vJoy Device` GUID
- Forces the RDP-safe `1024x768` windowed render mode
- Launches AC through the stock launcher
- Dismisses the setup wizard if needed
- Clicks through the stock launcher into Hotlap
- Enters cockpit view
- Clicks the in-session steering wheel icon so the car accepts drive input
- Can optionally skip the watcher with `-SkipWatcher`

Operator wrapper scripts:

- [Run-Game.ps1](../scripts/Run-Game.ps1)
- [Smoke-Test.ps1](../scripts/Smoke-Test.ps1)
- [Train-LiveAgent.ps1](../scripts/Train-LiveAgent.ps1)
- [Resume-LiveAgent.ps1](../scripts/Resume-LiveAgent.ps1)
- [Plot-Run.ps1](../scripts/Plot-Run.ps1)
- [Launch-TurningTrain.ps1](../scripts/Launch-TurningTrain.ps1)
- [Start-QwenRuntime.ps1](../scripts/Start-QwenRuntime.ps1)
- [Start-UnifiedDashboard.ps1](../scripts/Start-UnifiedDashboard.ps1)
- [Run-ExperimentSweep.ps1](../scripts/experimental/Run-ExperimentSweep.ps1)

## Typical Flow

Launch the game:

```powershell
powershell -ExecutionPolicy Bypass -File C:\Workspace\RacingSim\scripts\Run-Game.ps1 -SkipWatcher
```

Smoke test:

```powershell
powershell -ExecutionPolicy Bypass -File C:\Workspace\RacingSim\scripts\Smoke-Test.ps1
```

Short live SAC run:

```powershell
powershell -ExecutionPolicy Bypass -File C:\Workspace\RacingSim\scripts\Train-LiveAgent.ps1 -Algo sac -NumSteps 20000
```

Shared-backbone Qwen + SAC training:

```powershell
cd C:\Workspace\RacingSim\assetto_corsa_gym
uv run --no-sync python train.py --algo shared_sac --config config.yml
```

Resume training for a fixed time:

```powershell
powershell -ExecutionPolicy Bypass -File C:\Workspace\RacingSim\scripts\Resume-LiveAgent.ps1 -Algo sac -DurationHours 3
```

Start the dashboard:

```powershell
powershell -ExecutionPolicy Bypass -File C:\Workspace\RacingSim\scripts\Start-UnifiedDashboard.ps1
```

Plot a finished run:

```powershell
powershell -ExecutionPolicy Bypass -File C:\Workspace\RacingSim\scripts\Plot-Run.ps1 -RunDir C:\Workspace\RacingSim\assetto_corsa_gym\outputs\RUN_DIR
```

## Current Caveats

- Training is still noisy; short bad episodes and physics outliers occur
- The live launcher automation is pixel-position based against the stock AC launcher on this machine
- Planner latency is ~400ms p95 with the 2B model; still too high for tight control loops
- Launch-assist / reset handoff remains the main environment-side instability
- No run has completed a clean timed lap yet
- `plot_live_run.py` works but TensorBoard emits a harmless Windows path warning
