# Qwen + SAC Racing Agent: Complete Training Guide

> End-to-end documentation for the unified language-model backbone + Soft Actor-Critic
> reinforcement-learning system that learns to race in Assetto Corsa.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
   - [System Diagram](#21-system-diagram)
   - [Qwen Backbone & LoRA](#22-qwen-backbone--lora)
   - [State Tokenizer](#23-state-tokenizer)
   - [Temporal Compressor](#24-temporal-compressor)
   - [Plan Codes & Hindsight Labeling](#25-plan-codes--hindsight-labeling)
   - [FiLM-Conditioned SAC](#26-film-conditioned-sac)
   - [Student Distillation](#27-student-distillation)
3. [Training Pipeline](#3-training-pipeline)
   - [Data Flow](#31-data-flow)
   - [Replay Buffer](#32-replay-buffer)
   - [Loss Functions](#33-loss-functions)
   - [Hyperparameters](#34-hyperparameters)
4. [Assetto Corsa Setup](#4-assetto-corsa-setup)
   - [Prerequisites](#41-prerequisites)
   - [vJoy Installation](#42-vjoy-installation)
   - [Plugin Installation](#43-plugin-installation)
   - [AC Game Configuration](#44-ac-game-configuration)
   - [INI File Reference](#45-ini-file-reference)
5. [Running Training](#5-running-training)
   - [Quick Start](#51-quick-start)
   - [Launch Sequence](#52-launch-sequence)
   - [Monitoring](#53-monitoring)
   - [Checkpointing & Resume](#54-checkpointing--resume)
6. [File Reference](#6-file-reference)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Overview

This project trains a racing agent that combines a **large language model** (Qwen3.5-4B)
with **Soft Actor-Critic** reinforcement learning. The LLM backbone provides rich
intermediate representations (via a branch layer) that condition the RL policy through
FiLM (Feature-wise Linear Modulation). Both components learn together:

- **Qwen backbone** (LoRA fine-tuned) learns to encode driving situations into
  plan codes and latent representations.
- **SAC agent** learns continuous steering, throttle, and brake control conditioned
  on those representations.
- **Hindsight labeling** breaks the bootstrap cycle by computing ground-truth plan
  codes from actual episode outcomes.
- **Student distillation** trains a lightweight fallback encoder so inference can
  run without the full LLM when needed.

The agent drives inside **Assetto Corsa** via a UDP plugin that streams telemetry
and receives vJoy control inputs at 25 Hz.

---

## 2. Architecture

### 2.1 System Diagram

```
+-----------------------------------------------------------+
|  Assetto Corsa  (acs.exe)                                 |
|  +-- sensors_par plugin (Python 3.3)                      |
|  |   +-- ego_server.py      UDP :2345  (telemetry/ctrl)   |
|  |   +-- opponents_server   TCP :2346  (opponent data)    |
|  |   +-- sim_management     TCP :2347  (reset/info)       |
|  |   +-- car_control.py     vJoy interface                |
|  +--------------------------------------------------------+
|                       |  UDP 25 Hz
|                       v
|  +-- Python Gym Client (ac_client.py / ac_env.py) --------+
|  |   state (125-dim), reward, done, info                  |
|  +--------------------------------------------------------+
|                       |
|                       v
|  +-- SharedBackboneRuntime (unified_backbone.py) ---------+
|  |   Frame buffer (64 frames) --> StateTokenizer          |
|  |   --> TemporalCompressor (GRU + attention pooling)     |
|  |   --> Qwen3.5-4B backbone (branch @ layer 16)          |
|  |       +-- LoRA adapters (rank 16, alpha 32)            |
|  |       +-- PlanHead (6 plan-code fields)                |
|  |       +-- ValueHead, RiskHead                          |
|  |   --> z_mid (512-dim), plan_logits, confidence         |
|  +--------------------------------------------------------+
|                       |
|                       v
|  +-- SharedBackboneSAC (shared_backbone_sac.py) ----------+
|  |   PlanConditionEncoder (plan codes + numerics)         |
|  |   FiLM-conditioned Gaussian Policy  --> actions (3-dim)|
|  |   FiLM-conditioned Twin Q-networks  --> Q-values       |
|  |   Automatic entropy tuning (alpha)                     |
|  +--------------------------------------------------------+
|                       |
|                       v
|  +-- ReplayBuffer (6-tuple: s, a, r, s', done, s_prev) --+
|  |   Hindsight plan-code labels (6 fields per transition) |
|  |   N-step returns (n=3, gamma=0.992)                    |
|  +--------------------------------------------------------+
```

### 2.2 Qwen Backbone & LoRA

**File:** `planner/unified_backbone.py` -- class `BackboneWrapper`

The backbone loads a HuggingFace causal LM and extracts intermediate hidden states:

| Setting | Value | Description |
|---------|-------|-------------|
| `model_name` | `Qwen/Qwen3.5-4B-Base` | Primary model (4.2B params) |
| `fallback_model_name` | `Qwen/Qwen3.5-2B-Base` | Fallback if primary fails |
| `quantization` | `fp16` | Half-precision (or `4bit_nf4` for QLoRA) |
| `branch_layer_4b` | 16 | Layer to extract z_mid from (4B model) |
| `branch_layer_2b` | 12 | Layer to extract z_mid from (2B model) |
| `backbone_device_index` | 0 | GPU for the backbone |
| `head_device_index` | 1 | GPU for SAC policy/Q-networks |

**LoRA configuration:**

```python
LoraConfig(
    r=16,            # Rank
    lora_alpha=32,   # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Attention projections
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

The `forward()` method accepts an `enable_grad` flag. During inference it runs under
`torch.no_grad()`; during backbone training it enables gradients through LoRA parameters:

```python
context = nullcontext() if enable_grad else torch.no_grad()
with context:
    outputs = self._hf_model(inputs_embeds=inputs_embeds, ...)
```

**Gradient checkpointing** is enabled to reduce memory when training the 4B model.

### 2.3 State Tokenizer

**File:** `planner/unified_backbone.py` -- class `StateTokenizer`

Converts raw numeric observations into the backbone's embedding space:

| Projection | Input | Output | Purpose |
|-----------|-------|--------|---------|
| `obs_proj` | state_dim (125) | hidden_size (2560) | Main observation features |
| `delta_proj` | state_dim (125) | hidden_size (2560) | Frame-to-frame changes |
| `event_proj` | 6 | hidden_size (2560) | Binary driving events |
| `task_embedding` | task_id (int) | hidden_size (2560) | Track/car identity |

The 6 event bits are: `corner_entry`, `overspeed_event`, `gap_alert`,
`heading_alert`, `recovery_event`, `off_track_recent`.

Output: `LayerNorm(obs + delta + event + task)` with shape `[batch, steps, 2560]`.

### 2.4 Temporal Compressor

**File:** `planner/unified_backbone.py` -- class `TemporalCompressor`

Compresses a window of tokenized frames into a fixed set of summary tokens:

1. **GRU** (2-layer, hidden 512) processes the frame sequence.
2. **Attention pooling** uses 8 learnable query vectors to attend over GRU outputs.
3. Output: `summary_tokens` `[batch, 9, 2560]` (8 attention + 1 final-state token)
   and `state_summary` `[batch, 2560]` from the GRU hidden state.

### 2.5 Plan Codes & Hindsight Labeling

**File:** `planner/schemas.py` -- `PLAN_CODE_SCHEMA`

Six discrete driving-strategy fields provide structured conditioning:

| Field | Options | Description |
|-------|---------|-------------|
| `speed_mode` | conserve, nominal, push | Overall pace strategy |
| `brake_phase` | early, nominal, late, emergency | Braking timing |
| `line_mode` | tight, neutral, wide_exit, recovery_line | Racing line choice |
| `stability_mode` | neutral, rotate, stabilize | Car balance control |
| `recovery_mode` | off, on | Recovery from errors |
| `risk_mode` | low, medium, high | Risk tolerance |

**Hindsight labeling** (`planner/hindsight_labeler.py`) runs after each episode and
computes outcome-grounded plan codes from the actual results:

- Discounted future returns (identifies good vs bad segments)
- Offtrack-ahead flags (detects upcoming track excursions)
- Speed, curvature, and gap metrics

For example, `speed_mode="conserve"` is assigned when offtrack events follow, or
when the future return is below the 25th percentile for that episode.

This breaks the bootstrap cycle where the model would otherwise train on its own
(potentially incorrect) plan-code predictions.

### 2.6 FiLM-Conditioned SAC

**File:** `algorithm/discor/discor/algorithm/shared_backbone_sac.py`

The SAC policy and Q-networks are conditioned on backbone outputs via FiLM layers:

```
core = concat(z_mid[512], obs_proj(state)[128], delta_proj(delta)[128])  # 768-dim
condition = PlanConditionEncoder(plan_ids, z_mid, value_hat, confidence, offtrack_prob)

# FiLM modulation at each hidden layer:
h = LayerNorm((linear(h) * (1 + scale(condition))) + shift(condition))
```

**PlanConditionEncoder** maps:
- 6 plan-code embeddings (one per field)
- Numeric features: z_mid (projected), value_hat, confidence, offtrack_prob, valid flag
- Through a 2-layer MLP to produce the conditioning vector (~180-dim)

**Policy:** FiLM-conditioned Gaussian with tanh squashing, 3 action dimensions
(steer, throttle, brake). Initial biases:
- Mean: `[0.0, 0.35, -0.25]` (slight throttle bias, slight brake release)
- Log-std: `[-2.0, -1.0, -1.4]` (tighter steering, looser pedals)

**Twin Q-functions:** Two independent FiLM-conditioned networks for clipped double-Q.

### 2.7 Student Distillation

**File:** `planner/unified_backbone.py` -- class `StudentIntentEncoder`

A lightweight 2-layer MLP (state_dim -> 256 -> 256) that learns to approximate the
full backbone's `z_mid` output. Used as a fallback when backbone inference is too slow
or unavailable. Trained via MSE loss during `update_backbone()`:

```python
distill_loss = MSE(student_z_mid, backbone_z_mid.detach())
```

The student also predicts plan_logits, confidence, offtrack_prob, and value_hat.

---

## 3. Training Pipeline

### 3.1 Data Flow

Each training step follows this cycle:

1. **Observe** state from AC environment (125-dim vector at 25 Hz)
2. **Encode** through backbone runtime (buffered, cached at 5 Hz refresh)
3. **Sample action** from policy (or random during first 2000 steps)
4. **Execute** action in AC via vJoy
5. **Store** (state, action, reward, next_state, done, prev_state) in replay buffer
6. **Update** SAC networks every 4 environment steps
7. **Update backbone** (LoRA + heads) every 100 gradient steps
8. **Post-episode:** Run hindsight labeling on completed episode

### 3.2 Replay Buffer

**File:** `algorithm/discor/discor/replay_buffer.py`

Extended to support the unified backbone training:

- **6-tuple format:** `(state, action, reward, next_state, done, prev_state)`
- **N-step returns:** n=3, gamma=0.992 for temporal-difference targets
- **Hindsight labels:** 6-column int64 array storing plan-code IDs per transition
- **Memory:** 8M transitions (configurable)

The `prev_state` field enables computing actual state deltas during training,
rather than using zeros (which was a bug in the original wiring).

### 3.3 Loss Functions

The backbone optimizer minimizes four losses:

| Loss | Type | Target | Weight |
|------|------|--------|--------|
| **Value** | MSE | N-step discounted returns | 1.0 |
| **Risk** | BCE | Episode done signal | 1.0 |
| **Plan** | Cross-entropy | Hindsight plan-code labels (6 fields) | 1.0 |
| **Distill** | MSE | Student z_mid vs backbone z_mid | 0.5 |

Gradient clipping at max_norm=1.0. Two optimizer param groups:
- LoRA parameters: lr=1e-5 (conservative, preserves pretrained knowledge)
- Head parameters: lr=3e-4 (faster adaptation for task-specific heads)

SAC uses standard losses:
- **Policy:** Maximize Q(s,a) - alpha * log_pi(a|s)
- **Q-functions:** MSE on soft Bellman targets with clipped double-Q
- **Entropy:** Automatic alpha tuning toward target entropy

### 3.4 Hyperparameters

```yaml
# Backbone
model_name: Qwen/Qwen3.5-4B-Base
quantization: fp16
lora_rank: 16
lora_alpha: 32
lora_lr: 1e-5
head_lr: 3e-4
backbone_update_every: 100
student_distill_weight: 0.5
frame_buffer_len: 64
summary_token_count: 8
cache_refresh_hz: 5.0
branch_layer_4b: 16

# SAC
gamma: 0.992
nstep: 3
policy_lr: 3e-4
q_lr: 3e-4
entropy_lr: 3e-4
policy_hidden_units: [256, 256, 256]
q_hidden_units: [256, 256, 256]
target_update_coef: 0.005

# Agent
batch_size: 128
start_steps: 2000
update_every_steps: 4
post_episode_update_budget_s: 30
checkpoint_freq: 200000
```

---

## 4. Assetto Corsa Setup

### 4.1 Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Assetto Corsa | Steam version | With Content installed |
| Custom Shaders Patch | Latest | Required for car reset (`ext_resetCar`) |
| vJoy | 2.1.x | Virtual joystick driver |
| Python | 3.10+ | Training environment |
| PyTorch | 2.x + CUDA | GPU acceleration |
| peft | 0.13.x | LoRA adapter library |
| NVIDIA GPU(s) | 6GB+ VRAM each | Backbone on GPU 0, SAC on GPU 1 |

### 4.2 vJoy Installation

1. Download vJoy from <https://sourceforge.net/projects/vjoystick/>.
2. Install to `C:\Program Files\vJoy`.
3. Run **vJoyConf.exe** and configure **Device 1**:
   - Enable axes: X, Y, Z
   - Number of buttons: 2 (for gear shift)
   - All axes continuous, range 0-32768.
4. Verify with **JoyMonitor.exe** that axes respond.

The plugin uses `C:\Program Files\vJoy\x64\vJoyInterface.dll` to send inputs.

### 4.3 Plugin Installation

Copy the plugin from the repository to AC's apps folder:

```
Source:  <repo>/assetto_corsa_gym/AssettoCorsaPlugin/plugins/sensors_par/
Target:  C:\Program Files (x86)\Steam\steamapps\common\assettocorsa\apps\python\sensors_par\
```

Files to copy:
```
sensors_par.py        # Main plugin entry point
ego_server.py         # UDP server for agent communication
car_control.py        # vJoy control interface
config.py             # Server port configuration
sim_info.py           # AC shared memory reader
structures.py         # ctypes struct definitions
vjoy.py               # vJoy DLL wrapper
vjoy_linux.py         # Linux alternative (if applicable)
ac_logger.py          # Plugin logging
dual_buffer.py        # Shared memory double-buffer
precise_timer.py      # High-resolution timer
profiler.py           # Performance profiling
record_telemetry.py   # Telemetry recording
alternative_python.py # Screen capture subprocess spawner
screen_capture.py     # Screen capture module
screen_capture_worker.py  # Screen capture worker process
```

Also copy controller presets:
```
Source:  <repo>/assetto_corsa_gym/AssettoCorsaConfigs/controllers/savedsetups/
Target:  C:\Users\<user>\Documents\Assetto Corsa\cfg\controllers\savedsetups\
Files:   Vjoy.ini, WASD.ini
```

### 4.4 AC Game Configuration

#### Enable the Plugin

Edit (or verify) `<AC_ROOT>/cfg/python.ini`:

```ini
[SENSORS_PAR]
ACTIVE=1
```

Or enable in-game: Settings > General > UI Modules > sensors_par.

#### Set Controller to vJoy

Edit `<AC_ROOT>/cfg/controls.ini`:

```ini
[HEADER]
INPUT_METHOD=WHEEL

[CONTROLLERS]
CON0=vJoy Device
PGUID0=<your-vJoy-GUID>

[STEER]
JOY=0
AXLE=0
SCALE=1
LOCK=900
STEER_GAMMA=1
STEER_FILTER=0

[THROTTLE]
JOY=0
AXLE=1
MIN=-1
MAX=1

[BRAKES]
JOY=0
AXLE=2
MIN=-1
MAX=1
GAMMA=2.4

[CLUTCH]
JOY=-1
AXLE=-1

[GEARUP]
JOY=0
BUTTON=0

[GEARDN]
JOY=0
BUTTON=1
```

The PGUID can be found in Device Manager or vJoyConf. The axis mapping is:
- Axis 0 (X) = Steering
- Axis 1 (Y) = Throttle
- Axis 2 (Z) = Brake

#### Set Driving Assists

Edit `<AC_ROOT>/cfg/assists.ini`:

```ini
[ASSISTS]
IDEAL_LINE=0
AUTO_BLIP=1
STABILITY_CONTROL=0
AUTO_BRAKE=0
AUTO_SHIFTER=1          ; Must be ON for agent
ABS=0
TRACTION_CONTROL=0
AUTO_CLUTCH=1           ; Must be ON for agent
VISUALDAMAGE=0
DAMAGE=0
FUEL_RATE=0
TYRE_BLANKETS=0
```

Key: `AUTO_SHIFTER=1` and `AUTO_CLUTCH=1` are required since the agent only
controls steer/throttle/brake.

#### Set Video Frame Rate

Edit `<AC_ROOT>/cfg/video.ini` and set:
```ini
[VIDEO]
LIMIT=50    ; 50 FPS for local, 100 for remote
```

The plugin's main tick runs at 100 Hz but downsamples to `ego_sampling_freq` (25 Hz).

### 4.5 INI File Reference

#### race.ini

Located at `<AC_ROOT>/cfg/race.ini`. Must have a practice session defined:

```ini
[HEADER]
VERSION=1

[RACE]
TRACK=monza
CONFIG_TRACK=
MODEL=ks_mazda_miata
MODEL_CONFIG=
CARS=1
AI_LEVEL=98
FIXED_SETUP=0
PENALTIES=0

[SESSION_0]
NAME=Practice
TYPE=1
LAPS=0
TIME=0
SPAWN_SET=START

[GHOST_CAR]
RECORDING=1
PLAYING=1
SECONDS_ADVANTAGE=0
LOAD=1
FILE=

[REPLAY]
FILENAME=
ACTIVE=0

[LIGHTING]
SUN_ANGLE=-48
TIME_MULT=1
CLOUD_SPEED=0.2

[GROOVE]
VIRTUAL_LAPS=10
MAX_LAPS=30
STARTING_LAPS=0

[DYNAMIC_TRACK]
SESSION_START=100
SESSION_TRANSFER=50
RANDOMNESS=0
LAP_GAIN=1

[REMOTE]
ACTIVE=0

[LAP_INVALIDATOR]
ALLOWED_TYRES_OUT=-1

[TEMPERATURE]
AMBIENT=26
ROAD=32

[WEATHER]
NAME=4_mid_clear

[CAR_0]
MODEL=ks_mazda_miata
MODEL_CONFIG=
SKIN=00_classic_red
DRIVER_NAME=
NATIONALITY=ITA
AI_LEVEL=96

[AUTOSPAWN]
ACTIVE=1
```

Key points:
- `[SESSION_0]` with `TYPE=1` (Practice) is required for acs.exe direct launch.
- `TRACK=monza` matches the training config default.
- `CARS=1` for single-agent training (no opponents server needed).
- `[AUTOSPAWN] ACTIVE=1` auto-places the car on track.

#### Plugin config.py

Located at `<AC_ROOT>/apps/python/sensors_par/config.py`:

```python
class Config:
    ego_server_port = 2345              # UDP - main agent channel
    opponents_server_port = 2346        # TCP - opponent data
    simulation_management_server_port = 2347  # TCP - reset/info
    ego_sampling_freq = 25              # Hz observation rate
    sampling_freq = 100                 # Hz AC tick rate
    vjoy_executed_by_server = True      # Plugin writes vJoy (not client)
```

These ports must match `config.yml`:
```yaml
AssettoCorsa:
  ego_server_port: 2345
  opponents_server_port: 2346
  simulation_management_server_port: 2347
```

---

## 5. Running Training

### 5.1 Quick Start

```bash
# 1. Install Python dependencies
cd <repo>
uv sync          # or: pip install -e .

# 2. Install peft (must be compatible with your transformers version)
uv pip install peft==0.13.2

# 3. Launch Assetto Corsa (see 5.2 below)

# 4. Start training
.venv/Scripts/python.exe train.py --algo sac
```

### 5.2 Launch Sequence

#### Step 1: Start Assetto Corsa

Launch `acs.exe` directly (not the AC launcher):

```bash
cd "C:\Program Files (x86)\Steam\steamapps\common\assettocorsa"
start "" acs.exe
```

This reads `cfg/race.ini` and loads directly into the configured session. Ensure:
- `[SESSION_0]` section exists with `TYPE=1` (Practice).
- `[AUTOSPAWN] ACTIVE=1` is set.
- The sensors_par plugin is enabled in `cfg/python.ini`.

Wait ~20 seconds for the game to fully load onto the track.

#### Step 2: Verify Servers

Check that the plugin servers are running:

```bash
# Sim management (TCP)
python -c "
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(5)
s.connect(('localhost', 2347))
s.sendall(b'get_static_info')
data = s.recv(1048576).decode()
s.close()
print('Sim management OK:', data[:100])
"

# Ego server (UDP)
python -c "
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.settimeout(3)
s.sendto(b'connect', ('localhost', 2345))
data, _ = s.recvfrom(1024)
s.close()
print('Ego server response:', data.decode())
"
```

Expected: sim management returns track info; ego server returns `identified`.

#### Step 3: Run Training

```bash
.venv/Scripts/python.exe train.py --algo sac
```

The training script:
1. Loads `config.yml`
2. Creates the AC gym environment
3. Detects `UnifiedBackbone.enabled=True` and instantiates `SharedBackboneSAC`
4. Loads Qwen3.5-4B, enables LoRA and gradient checkpointing
5. Creates the backbone optimizer (LoRA lr=1e-5, heads lr=3e-4)
6. Begins the training loop

To disable wandb logging (if not configured):
```yaml
# In config.yml
disable_wandb: True
```

### 5.3 Monitoring

Training outputs appear in `outputs/<timestamp>/`:

```
outputs/
  20260314_115158/
    model/
      final/
        lora_adapters/     # LoRA checkpoint
        policy.pth         # SAC policy weights
        q1.pth, q2.pth    # Q-network weights
    laps/
      *.parquet            # Per-episode state logs
    train.csv              # Training metrics
```

Key log lines to watch:

```
# Model loaded successfully
planner.unified_backbone Loaded backbone model Qwen/Qwen3.5-4B-Base on cuda:0
planner.unified_backbone LoRA enabled: 1835008 trainable / 4207586304 total params

# Episode completed
ac_env total_steps: 1237 ep_steps: 251 ep_reward: 9.3 LapDist: 5107.69

# Training throughput
agent Episode throughput. env_steps/s: 7.66 gradient_updates/s: 0.07

# Backbone latency (informational)
planner.unified_backbone Planner latency p95 426.30 ms exceeds threshold
```

### 5.4 Checkpointing & Resume

Checkpoints are saved every `checkpoint_freq` steps (default 200,000).
The model also saves on clean exit or crash recovery.

LoRA adapters are saved separately:
```python
# Save
backbone.save_lora("path/to/lora_adapters")

# Load
backbone.load_lora("path/to/lora_adapters")
```

---

## 6. File Reference

### Core Architecture

| File | Description |
|------|-------------|
| `planner/unified_backbone.py` | Qwen backbone, tokenizer, compressor, encoder, student, runtime |
| `planner/schemas.py` | Plan code schema (6 fields), conversion utilities |
| `planner/hindsight_labeler.py` | Post-episode outcome-grounded plan-code labeling |

### RL Algorithm

| File | Description |
|------|-------------|
| `algorithm/discor/discor/algorithm/shared_backbone_sac.py` | FiLM-conditioned SAC with backbone training |
| `algorithm/discor/discor/algorithm/sac.py` | Base SAC implementation |
| `algorithm/discor/discor/algorithm/discor.py` | DisCor error-model extension |
| `algorithm/discor/discor/replay_buffer.py` | Extended replay buffer (prev_state, hindsight) |
| `algorithm/discor/discor/agent.py` | Training loop, episode lifecycle, hindsight integration |

### Environment

| File | Description |
|------|-------------|
| `assetto_corsa_gym/AssettoCorsaEnv/ac_env.py` | Gym environment wrapping AC |
| `assetto_corsa_gym/AssettoCorsaEnv/ac_client.py` | UDP/TCP client for AC servers |

### AC Plugin (runs inside Assetto Corsa)

| File | Description |
|------|-------------|
| `assetto_corsa_gym/AssettoCorsaPlugin/plugins/sensors_par/sensors_par.py` | Plugin entry, server threads |
| `assetto_corsa_gym/AssettoCorsaPlugin/plugins/sensors_par/ego_server.py` | UDP ego server (port 2345) |
| `assetto_corsa_gym/AssettoCorsaPlugin/plugins/sensors_par/car_control.py` | vJoy control writer |
| `assetto_corsa_gym/AssettoCorsaPlugin/plugins/sensors_par/config.py` | Port/frequency configuration |

### Configuration

| File | Description |
|------|-------------|
| `config.yml` | Main training configuration |
| `train.py` | Training entry point |

---

## 7. Troubleshooting

### Plugin servers not starting

**Symptom:** Port 2347 (sim management) responds but port 2345 (ego server, UDP)
does not bind.

**Cause:** AC was launched without a `[SESSION_0]` section in `race.ini`, or the
game is sitting at a menu instead of on-track.

**Fix:** Ensure `race.ini` has `[SESSION_0]` with `TYPE=1` and `[AUTOSPAWN] ACTIVE=1`.
Kill `acs.exe` and relaunch. Verify with `netstat -ano | findstr 2345` that UDP port
is bound.

### Car not moving (accStatus stays 0)

**Symptom:** Launch assist logs show `acc=0.000` despite commanding throttle.

**Cause:** vJoy device not acquired by the plugin, or AC is not reading from the
vJoy controller.

**Fix:**
1. Verify vJoy device is configured in `controls.ini` as `CON0=vJoy Device`.
2. Check that no other application has acquired vJoy Device 1.
3. The plugin auto-recovers by relaunching the AC session after a stuck reset.

### ConnectionRefusedError on port 2345

**Symptom:** `[WinError 10061] No connection could be made`.

**Cause:** AC is not running, or the plugin failed to start the ego server.

**Fix:** Launch `acs.exe` and wait for full track load (~20s). Check that the
`sensors_par` plugin is enabled in `cfg/python.ini`.

### peft ImportError (HybridCache)

**Symptom:** `ImportError: cannot import name 'HybridCache' from 'transformers'`.

**Cause:** Version mismatch between peft and transformers.

**Fix:** Install a compatible peft version:
```bash
uv pip install peft==0.13.2
```

### wandb login error

**Symptom:** `UsageError: No API key configured`.

**Fix:** Either run `wandb login` or disable wandb in `config.yml`:
```yaml
disable_wandb: True
```

### High backbone latency warnings

**Symptom:** Repeated `Planner latency p95 ~430ms exceeds threshold` warnings.

**Cause:** Qwen3.5-4B inference on a consumer GPU (e.g., GTX 1660) is ~400ms per
call, exceeding the 5 Hz target.

**Mitigation:** This is informational only. Training proceeds normally. The backbone
cache is refreshed at the achievable rate. For faster inference, use:
- `quantization: 4bit_nf4` (halves VRAM, ~30% faster)
- `fallback_model_name: Qwen/Qwen3.5-2B-Base` (smaller model)
- The student encoder provides inference without backbone calls once distilled.

### Episode terminates immediately

**Symptom:** Very short episodes (< 10 steps) with `is_out_of_track`.

**Cause:** Early exploration with random actions quickly drives off track.

**Expected:** Normal during the first few hundred episodes. The agent improves
as the replay buffer fills and SAC updates begin (after `start_steps=2000`).
