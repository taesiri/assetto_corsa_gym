from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
except Exception:  # pragma: no cover
    BitsAndBytesConfig = None

try:
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
except Exception:  # pragma: no cover
    LoraConfig = None
    TaskType = None
    get_peft_model = None
    prepare_model_for_kbit_training = None


DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass
class LoadedQwen:
    model_name: str
    model: Any
    tokenizer: Any
    config: Any
    device: str
    load_mode: str
    quantized_4bit: bool
    fallback_used: bool


def model_dimensions(config: Any) -> tuple[int, int]:
    text_config = getattr(config, "text_config", None)
    hidden_size = getattr(config, "hidden_size", None)
    num_hidden_layers = getattr(config, "num_hidden_layers", None)
    if hidden_size is None and text_config is not None:
        hidden_size = getattr(text_config, "hidden_size", 0)
    if num_hidden_layers is None and text_config is not None:
        num_hidden_layers = getattr(text_config, "num_hidden_layers", 0)
    return int(hidden_size or 0), int(num_hidden_layers or 0)


def build_load_kwargs(device: str, load_mode: str) -> tuple[dict[str, Any], str, bool]:
    requested_mode = load_mode
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    quantized_4bit = False
    if device.startswith("cuda"):
        if load_mode in {"4bit", "auto"} and BitsAndBytesConfig is not None:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            kwargs["device_map"] = {"": device}
            quantized_4bit = True
            requested_mode = "4bit"
        else:
            kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"] = {"": device}
            requested_mode = "fp16"
    else:
        kwargs["torch_dtype"] = torch.float32
        requested_mode = "cpu"
    return kwargs, requested_mode, quantized_4bit


def load_qwen(
    *,
    model_name: str,
    fallback_model_name: str | None = None,
    device: str = "cuda:0",
    load_mode: str = "auto",
) -> LoadedQwen:
    tried: list[tuple[str, Exception]] = []
    candidates = [model_name]
    if fallback_model_name and fallback_model_name != model_name:
        candidates.append(fallback_model_name)

    for index, candidate in enumerate(candidates):
        try:
            config = AutoConfig.from_pretrained(candidate, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(candidate, trust_remote_code=True)
            kwargs, effective_mode, quantized_4bit = build_load_kwargs(device=device, load_mode=load_mode)
            model = AutoModelForCausalLM.from_pretrained(candidate, **kwargs)
            if device.startswith("cuda") and not quantized_4bit and "device_map" not in kwargs:
                model.to(device)
            model.eval()
            return LoadedQwen(
                model_name=candidate,
                model=model,
                tokenizer=tokenizer,
                config=config,
                device=device,
                load_mode=effective_mode,
                quantized_4bit=quantized_4bit,
                fallback_used=index > 0,
            )
        except Exception as exc:  # pragma: no cover - exercised in runtime
            tried.append((candidate, exc))

    details = "; ".join(f"{name}: {type(exc).__name__}: {exc}" for name, exc in tried)
    raise RuntimeError(f"Unable to load any Qwen candidate. {details}")


def attach_lora(
    loaded: LoadedQwen,
    *,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> Any:
    if LoraConfig is None or get_peft_model is None or TaskType is None:
        raise RuntimeError("peft is not installed in the Qwen runtime environment.")

    target_modules = target_modules or list(DEFAULT_TARGET_MODULES)
    model = loaded.model
    if loaded.quantized_4bit and prepare_model_for_kbit_training is not None:
        model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=int(lora_r),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    return get_peft_model(model, config)


def count_parameters(model: Any) -> dict[str, int]:
    total = 0
    trainable = 0
    for parameter in model.parameters():
        count = int(parameter.numel())
        total += count
        if parameter.requires_grad:
            trainable += count
    return {"total_params": total, "trainable_params": trainable}


def readiness_report(
    *,
    model_name: str,
    fallback_model_name: str | None,
    device: str,
    load_mode: str,
    attach_trainable_lora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> dict[str, Any]:
    loaded = load_qwen(
        model_name=model_name,
        fallback_model_name=fallback_model_name,
        device=device,
        load_mode=load_mode,
    )
    trainable_model = loaded.model
    lora_enabled = False
    if attach_trainable_lora:
        trainable_model = attach_lora(
            loaded,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        lora_enabled = True

    params = count_parameters(trainable_model)
    report = {
        "requested_model_name": model_name,
        "loaded_model_name": loaded.model_name,
        "fallback_used": loaded.fallback_used,
        "device": loaded.device,
        "load_mode": loaded.load_mode,
        "quantized_4bit": loaded.quantized_4bit,
        "model_type": getattr(loaded.config, "model_type", None),
        "hidden_size": model_dimensions(loaded.config)[0],
        "num_hidden_layers": model_dimensions(loaded.config)[1],
        "torch_dtype": str(getattr(trainable_model, "dtype", getattr(loaded.model, "dtype", None))),
        "lora_enabled": lora_enabled,
        "lora_r": int(lora_r) if lora_enabled else 0,
        "lora_alpha": int(lora_alpha) if lora_enabled else 0,
        "lora_dropout": float(lora_dropout) if lora_enabled else 0.0,
    }
    report.update(params)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify that a real Qwen 3.5 model can load and be made trainable.")
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B-Base")
    parser.add_argument("--fallback-model-name", default="Qwen/Qwen3.5-2B-Base")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--load-mode", choices=["auto", "4bit", "fp16", "cpu"], default="auto")
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = readiness_report(
        model_name=args.model_name,
        fallback_model_name=args.fallback_model_name,
        device=args.device,
        load_mode=args.load_mode,
        attach_trainable_lora=not args.no_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
