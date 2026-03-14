# Qwen Runtime

This project is a dedicated `uv` Python `3.10` runtime for loading `Qwen/Qwen3.5-*` models without changing the main simulator stack, which remains pinned to Python `3.9`.

## Why it exists

- The main training repo is pinned to Python `3.9`.
- `Qwen/Qwen3.5-*` requires a newer `transformers` build that now expects Python `3.10+`.
- This sidecar keeps the simulator stable while allowing real Qwen loading and LoRA readiness checks.

## Commands

Install and sync:

```powershell
uv sync --python 3.10
```

Verify that Qwen loads and can be wrapped with trainable LoRA adapters:

```powershell
uv run --python 3.10 python qwen_loader.py --model-name Qwen/Qwen3.5-4B-Base --fallback-model-name Qwen/Qwen3.5-2B-Base --device cuda:0 --load-mode fp16
```

Start the resident runtime:

```powershell
uv run --python 3.10 python qwen_sidecar.py --model-name Qwen/Qwen3.5-4B-Base --fallback-model-name Qwen/Qwen3.5-2B-Base --device cuda:0 --load-mode fp16 --bind 127.0.0.1 --port 8092
```
