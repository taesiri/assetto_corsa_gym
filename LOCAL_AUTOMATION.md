# Local Automation Notes

This checkout has local scripts and code changes for running AssettoCorsaGym unattended on Windows against the stock Assetto Corsa launcher.

## Added Scripts

- [smoke_random.py](C:/Workspace/RacingSim/assetto_corsa_gym/smoke_random.py)
  - Live wiring smoke test against the running sim.
- [live_train_watch.ps1](C:/Workspace/RacingSim/assetto_corsa_gym/live_train_watch.ps1)
  - Waits for the plugin TCP port and then runs the smoke test followed by a short SAC job.
- [plot_live_run.py](C:/Workspace/RacingSim/assetto_corsa_gym/plot_live_run.py)
  - Exports `live_training_metrics.png`, `live_trajectory_overlay.png`, `live_scalars.csv`, and `live_run_summary.json` for a finished live run directory.
- [train_for_duration.py](C:/Workspace/RacingSim/assetto_corsa_gym/train_for_duration.py)
  - Trains for a fixed wall-clock duration, can resume from a saved model, can seed the replay buffer from prior lap files, and saves `replay_buffer.pkl` with the final checkpoint.
- [offline_pretrain.py](C:/Workspace/RacingSim/assetto_corsa_gym/offline_pretrain.py)
  - Short offline SAC pretrain against the public Hugging Face dataset with metrics and overlay plots.

## Local Code Changes

- [ac_env.py](C:/Workspace/RacingSim/assetto_corsa_gym/assetto_corsa_gym/AssettoCorsaEnv/ac_env.py)
  - Added post-reset launch assist. If AC respawns in neutral, the wrapper now holds throttle through vJoy until the automatic gearbox engages and the car starts rolling. If that fails, it falls back to the outer launcher script.
- [sensors_par.py](C:/Workspace/RacingSim/assetto_corsa_gym/assetto_corsa_gym/AssettoCorsaPlugin/plugins/sensors_par/sensors_par.py)
  - Added guards so the plugin can run even if CSP helper APIs are missing.
- [WASD.ini](C:/Workspace/RacingSim/assetto_corsa_gym/assetto_corsa_gym/AssettoCorsaPlugin/windows-libs/WASD.ini)
  - Removed stale vJoy bindings from the keyboard profile.

## Outer Launcher Script

The stock-launcher automation lives outside this checkout at:

- [start_acgym_supported.ps1](C:/Workspace/RacingSim/start_acgym_supported.ps1)

That script:

- detects the real AC config root under Documents / OneDrive,
- forces `monza` + `ks_mazda_miata` Hotlap,
- enables `sensors_par`,
- copies the `Vjoy.ini` preset into the active `controls.ini`,
- launches AC through Steam,
- dismisses the setup wizard if needed,
- clicks through the stock launcher,
- enters cockpit view,
- can optionally skip the watcher with `-SkipWatcher`.

## Typical Flow

Launch the game:

```powershell
powershell -ExecutionPolicy Bypass -File C:\Workspace\RacingSim\start_acgym_supported.ps1
```

Smoke test:

```powershell
C:\Workspace\RacingSim\.venv-acgym\Scripts\python.exe C:\Workspace\RacingSim\assetto_corsa_gym\smoke_random.py
```

Short live SAC run:

```powershell
C:\Workspace\RacingSim\.venv-acgym\Scripts\python.exe C:\Workspace\RacingSim\assetto_corsa_gym\train.py disable_wandb=True Agent.num_steps=20000 Agent.memory_size=50000 Agent.offline_buffer_size=0 Agent.start_steps=1000 Agent.batch_size=64 AssettoCorsa.track=monza AssettoCorsa.car=ks_mazda_miata
```

Plot a finished live run:

```powershell
C:\Workspace\RacingSim\.venv-acgym\Scripts\python.exe C:\Workspace\RacingSim\assetto_corsa_gym\plot_live_run.py --run-dir C:\Workspace\RacingSim\assetto_corsa_gym\outputs\RUN_DIR
```

Continue training for a fixed time budget from an existing model:

```powershell
C:\Workspace\RacingSim\.venv-acgym\Scripts\python.exe C:\Workspace\RacingSim\assetto_corsa_gym\train_for_duration.py --load_path C:\Workspace\RacingSim\assetto_corsa_gym\outputs\RUN_DIR\model\final --duration-hours 3 --seed-laps-dir C:\Workspace\RacingSim\assetto_corsa_gym\outputs\RUN_DIR\laps disable_wandb=True AssettoCorsa.track=monza AssettoCorsa.car=ks_mazda_miata
```

## Verified Run Outputs

- Short online sanity run:
  - [20260311_125435.333](C:/Workspace/RacingSim/assetto_corsa_gym/outputs/20260311_125435.333)
- Full 20k-step live SAC run:
  - [20260311_125752.155](C:/Workspace/RacingSim/assetto_corsa_gym/outputs/20260311_125752.155)
- 3-hour continuation run with replay buffer saved:
  - [20260311_132637.378_duration](C:/Workspace/RacingSim/assetto_corsa_gym/outputs/20260311_132637.378_duration)

## Current Caveats

- The setup is operational, but training is still noisy and can produce short bad episodes or physics outliers.
- The live launcher automation is pixel-position based against the stock Assetto Corsa launcher on this machine, not a general menu automation framework.
- `plot_live_run.py` works, but TensorBoard emits a harmless Windows path warning while loading the scalar file.
