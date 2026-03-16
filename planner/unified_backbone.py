from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
import logging
import math
import threading
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from .schemas import DEFAULT_PLAN_CODE, OBS_CHANNEL_NAMES_DEFAULT, PLAN_CODE_SCHEMA, UnifiedBackboneOutput, canonicalize_plan_code, plan_code_from_ids

logger = logging.getLogger(__name__)


def _safe_torch_device(preferred: str | torch.device | None) -> torch.device:
    if preferred is None:
        return torch.device("cpu")
    if isinstance(preferred, torch.device):
        if preferred.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return preferred
    device = torch.device(preferred)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def _config_attr(config: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    nested = getattr(config, "text_config", None)
    if nested is not None:
        for name in names:
            value = getattr(nested, name, None)
            if value is not None:
                return value
    return default


class StateTokenizer(nn.Module):
    def __init__(self, state_dim: int, hidden_size: int, *, event_dim: int = 6, max_tasks: int = 16) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.hidden_size = int(hidden_size)
        self.event_dim = int(event_dim)
        self.max_tasks = max(1, int(max_tasks))
        self.obs_proj = nn.Sequential(nn.Linear(self.state_dim, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.delta_proj = nn.Sequential(nn.Linear(self.state_dim, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.event_proj = nn.Sequential(nn.Linear(self.event_dim, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.task_embedding = nn.Embedding(self.max_tasks, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        obs_seq: torch.Tensor,
        delta_seq: torch.Tensor | None = None,
        task_ids: torch.Tensor | None = None,
        event_bits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if obs_seq.dim() != 3:
            raise ValueError("obs_seq must have shape [batch, steps, state_dim]")
        batch_size, steps, _ = obs_seq.shape
        delta_seq = torch.zeros_like(obs_seq) if delta_seq is None else delta_seq
        event_bits = (
            torch.zeros(batch_size, steps, self.event_dim, device=obs_seq.device, dtype=obs_seq.dtype)
            if event_bits is None else event_bits
        )
        if task_ids is None:
            task_ids = torch.zeros(batch_size, steps, device=obs_seq.device, dtype=torch.long)
        elif task_ids.dim() == 1:
            task_ids = task_ids[:, None].expand(batch_size, steps)

        flat_obs = obs_seq.reshape(batch_size * steps, -1)
        flat_delta = delta_seq.reshape(batch_size * steps, -1)
        flat_event_bits = event_bits.reshape(batch_size * steps, -1)
        flat_task_ids = task_ids.reshape(batch_size * steps).clamp(min=0, max=self.max_tasks - 1)

        token = (
            self.obs_proj(flat_obs)
            + self.delta_proj(flat_delta)
            + self.event_proj(flat_event_bits)
            + self.task_embedding(flat_task_ids)
        )
        token = self.norm(token)
        return token.view(batch_size, steps, self.hidden_size)


class TemporalCompressor(nn.Module):
    def __init__(self, hidden_size: int, *, summary_token_count: int = 8, gru_hidden_size: int = 512, gru_layers: int = 2) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.summary_token_count = int(summary_token_count)
        self.gru = nn.GRU(hidden_size, gru_hidden_size, num_layers=gru_layers, batch_first=True)
        self.query = nn.Parameter(torch.randn(self.summary_token_count, gru_hidden_size) / math.sqrt(gru_hidden_size))
        self.summary_proj = nn.Linear(gru_hidden_size, hidden_size)
        self.state_summary_proj = nn.Linear(gru_hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, frame_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if frame_tokens.dim() != 3:
            raise ValueError("frame_tokens must have shape [batch, steps, hidden_size]")
        seq_out, hidden = self.gru(frame_tokens)
        attn_scores = torch.einsum("qd,bsd->bqs", self.query, seq_out) / math.sqrt(seq_out.shape[-1])
        attn_weights = torch.softmax(attn_scores, dim=-1)
        pooled = torch.einsum("bqs,bsd->bqd", attn_weights, seq_out)
        summary_tokens = self.summary_proj(pooled)
        state_summary = self.state_summary_proj(hidden[-1])
        summary_tokens = torch.cat([summary_tokens, state_summary[:, None, :]], dim=1)
        return self.norm(summary_tokens), state_summary


class FallbackBackboneModel(nn.Module):
    def __init__(self, hidden_size: int, *, layers: int = 6, heads: int = 8) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=hidden_size * 4,
            batch_first=True,
            dropout=0.0,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, inputs_embeds: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        hidden_states = [inputs_embeds]
        current = inputs_embeds
        for layer in self.encoder.layers:
            current = layer(current)
            hidden_states.append(current)
        current = self.norm(current)
        hidden_states[-1] = current
        return hidden_states, current


class BackboneWrapper(nn.Module):
    def __init__(
        self,
        model_name: str,
        fallback_model_name: str,
        *,
        device: torch.device,
        branch_layer_4b: int = 16,
        branch_layer_2b: int = 12,
        quantization: str = "4bit_nf4",
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.fallback_model_name = fallback_model_name
        self.quantization = quantization
        self.device = _safe_torch_device(device)
        self.branch_layer_4b = int(branch_layer_4b)
        self.branch_layer_2b = int(branch_layer_2b)
        self.hidden_size = 512
        self.num_layers = 6
        self.active_model_name = "internal_fallback"
        self._tokenizer = None
        self._hf_model = None
        self._can_generate = False
        self._lora_enabled = False
        self.fallback_model = None
        self._load_model(model_name=self.model_name)

    @property
    def can_generate(self) -> bool:
        return self._can_generate

    @property
    def branch_layer(self) -> int:
        if self.hidden_size >= 2560:
            return min(self.branch_layer_4b, self.num_layers)
        return min(self.branch_layer_2b, self.num_layers)

    def _activate_internal_fallback(self, *, reason: str) -> None:
        self._hf_model = None
        self._tokenizer = None
        self._can_generate = False
        self.hidden_size = 512
        self.num_layers = 6
        self.active_model_name = "internal_fallback"
        self.fallback_model = FallbackBackboneModel(self.hidden_size).to(self.device)
        self.fallback_model.eval()
        logger.warning("Using internal fallback backbone. reason=%s", reason)

    def _try_load_hf_model(self, model_name: str) -> bool:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            try:
                from transformers import BitsAndBytesConfig
            except Exception:
                BitsAndBytesConfig = None

            kwargs: dict[str, Any] = {"trust_remote_code": True}
            using_device_map = False
            if self.device.type == "cuda":
                kwargs["torch_dtype"] = torch.float16
            if self.quantization == "4bit_nf4" and BitsAndBytesConfig is not None and self.device.type == "cuda":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                kwargs["device_map"] = {"": str(self.device)}
                using_device_map = True

            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
            if not using_device_map:
                model.to(self.device)
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

            hidden_size = int(_config_attr(model.config, "hidden_size", "d_model", default=2048))
            num_layers = int(_config_attr(model.config, "num_hidden_layers", "n_layer", default=24))

            self._hf_model = model
            self._tokenizer = tokenizer
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.active_model_name = model_name
            self._can_generate = True
            self.fallback_model = None
            logger.info(
                "Loaded backbone model %s on %s (hidden_size=%d, layers=%d, quantization=%s)",
                model_name,
                self.device,
                self.hidden_size,
                self.num_layers,
                self.quantization,
            )
            return True
        except Exception as exc:
            logger.exception("Failed to load backbone model %s: %s", model_name, exc)
            return False

    def _load_model(self, model_name: str) -> None:
        if self._try_load_hf_model(model_name):
            return
        if model_name != self.fallback_model_name and self._try_load_hf_model(self.fallback_model_name):
            return
        self._activate_internal_fallback(reason=f"failed_models={model_name},{self.fallback_model_name}")

    def maybe_switch_to_fallback(self) -> None:
        if self.active_model_name == self.fallback_model_name:
            return
        logger.warning("Switching backbone from %s to configured fallback model %s", self.active_model_name, self.fallback_model_name)
        if self._try_load_hf_model(self.fallback_model_name):
            return
        self._activate_internal_fallback(reason=f"latency_or_reliability_fallback={self.fallback_model_name}")

    def enable_lora(self, *, rank: int = 16, alpha: int = 32, target_modules: list[str] | None = None, dropout: float = 0.05) -> bool:
        if self._hf_model is None:
            logger.warning("Cannot enable LoRA: no HF model loaded")
            return False
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            logger.warning("Cannot enable LoRA: peft not installed")
            return False
        if self._lora_enabled:
            logger.info("LoRA already enabled, skipping")
            return True
        target_modules = target_modules or ["q_proj", "v_proj"]
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self._hf_model = get_peft_model(self._hf_model, lora_config)
        self._lora_enabled = True
        trainable = sum(p.numel() for p in self._hf_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self._hf_model.parameters())
        logger.info("LoRA enabled: %d trainable / %d total params (%.2f%%)", trainable, total, 100.0 * trainable / max(total, 1))
        return True

    def enable_gradient_checkpointing(self) -> None:
        if self._hf_model is None:
            return
        try:
            model = self._hf_model.base_model if self._lora_enabled else self._hf_model
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled on backbone")
        except Exception as exc:
            logger.warning("Failed to enable gradient checkpointing: %s", exc)

    def lora_parameters(self) -> list[nn.Parameter]:
        if not self._lora_enabled or self._hf_model is None:
            return []
        return [p for p in self._hf_model.parameters() if p.requires_grad]

    def save_lora(self, path: str) -> None:
        if not self._lora_enabled or self._hf_model is None:
            logger.warning("Cannot save LoRA: not enabled or no model")
            return
        self._hf_model.save_pretrained(path)
        logger.info("LoRA adapters saved to %s", path)

    def load_lora(self, path: str) -> None:
        if self._hf_model is None:
            logger.warning("Cannot load LoRA: no HF model")
            return
        try:
            from peft import PeftModel
            if not self._lora_enabled:
                self._hf_model = PeftModel.from_pretrained(self._hf_model, path)
                self._lora_enabled = True
            else:
                from peft import set_peft_model_state_dict
                import torch as _torch
                adapters_weights = _torch.load(path + "/adapter_model.bin", map_location=self.device)
                set_peft_model_state_dict(self._hf_model, adapters_weights)
            logger.info("LoRA adapters loaded from %s", path)
        except Exception as exc:
            logger.exception("Failed to load LoRA adapters: %s", exc)

    def forward(self, inputs_embeds: torch.Tensor, require_top: bool = False, *, enable_grad: bool = False) -> dict[str, torch.Tensor]:
        inputs_embeds = inputs_embeds.to(self.device)
        if self._hf_model is not None:
            model_dtype = next(self._hf_model.parameters()).dtype
            inputs_embeds = inputs_embeds.to(dtype=model_dtype)
            context = nullcontext() if enable_grad else torch.no_grad()
            with context:
                outputs = self._hf_model(
                    inputs_embeds=inputs_embeds,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
            all_hidden = outputs.hidden_states
            branch_hidden = all_hidden[self.branch_layer][:, -1, :]
            top_hidden = all_hidden[-1][:, -1, :] if require_top else branch_hidden
            return {"branch_hidden": branch_hidden, "top_hidden": top_hidden}

        hidden_states, last_hidden = self.fallback_model(inputs_embeds)
        branch_hidden = hidden_states[min(self.branch_layer, len(hidden_states) - 1)][:, -1, :]
        top_hidden = last_hidden[:, -1, :] if require_top else branch_hidden
        return {"branch_hidden": branch_hidden, "top_hidden": top_hidden}

    def generate_text(self, prompt: str, *, max_new_tokens: int = 96) -> str | None:
        if not self._can_generate or self._tokenizer is None or self._hf_model is None:
            return None
        try:
            encoded = self._tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                generated = self._hf_model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            text = self._tokenizer.decode(generated[0], skip_special_tokens=True)
            return text[len(prompt):].strip() if text.startswith(prompt) else text.strip()
        except Exception as exc:
            logger.exception("Backbone text generation failed: %s", exc)
            return None


class UnifiedBackboneEncoder(nn.Module):
    def __init__(
        self,
        state_dim: int,
        *,
        model_name: str,
        fallback_model_name: str,
        device: torch.device,
        max_tasks: int,
        summary_token_count: int,
        quantization: str,
        branch_layer_4b: int,
        branch_layer_2b: int,
    ) -> None:
        super().__init__()
        self.device = _safe_torch_device(device)
        self.backbone = BackboneWrapper(
            model_name=model_name,
            fallback_model_name=fallback_model_name,
            device=self.device,
            branch_layer_4b=branch_layer_4b,
            branch_layer_2b=branch_layer_2b,
            quantization=quantization,
        )
        self.state_tokenizer = StateTokenizer(state_dim=state_dim, hidden_size=self.backbone.hidden_size, max_tasks=max_tasks)
        self.temporal_compressor = TemporalCompressor(hidden_size=self.backbone.hidden_size, summary_token_count=summary_token_count)
        self.state_tokenizer.to(self.device)
        self.temporal_compressor.to(self.device)

    def forward(
        self,
        obs_seq: torch.Tensor,
        *,
        delta_seq: torch.Tensor | None = None,
        task_ids: torch.Tensor | None = None,
        event_bits: torch.Tensor | None = None,
        require_top: bool = False,
        enable_grad: bool = False,
    ) -> dict[str, torch.Tensor]:
        obs_seq = obs_seq.to(self.device)
        delta_seq = None if delta_seq is None else delta_seq.to(self.device)
        task_ids = None if task_ids is None else task_ids.to(self.device)
        event_bits = None if event_bits is None else event_bits.to(self.device)
        frame_tokens = self.state_tokenizer(obs_seq, delta_seq=delta_seq, task_ids=task_ids, event_bits=event_bits)
        summary_tokens, state_summary = self.temporal_compressor(frame_tokens)
        backbone_outputs = self.backbone(summary_tokens, require_top=require_top, enable_grad=enable_grad)
        z_mid = backbone_outputs["branch_hidden"].float()
        top_hidden = backbone_outputs["top_hidden"].float()
        return {
            "summary_tokens": summary_tokens,
            "state_summary": state_summary,
            "z_mid": z_mid,
            "top_hidden": top_hidden,
        }


class PlanHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.heads = nn.ModuleDict({
            field_name: nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, len(options)),
            )
            for field_name, options in PLAN_CODE_SCHEMA.items()
        })

    def forward(self, z_mid: torch.Tensor) -> dict[str, torch.Tensor]:
        return {field_name: head(z_mid) for field_name, head in self.heads.items()}


class ValueHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, 1))

    def forward(self, z_mid: torch.Tensor) -> torch.Tensor:
        return self.net(z_mid)


class RiskHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, 2))

    def forward(self, z_mid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(z_mid)
        return torch.sigmoid(logits[:, :1]), torch.sigmoid(logits[:, 1:])


class SegmentValueNet(nn.Module):
    def __init__(self, input_dim: int, hidden_units: list[int] | None = None) -> None:
        super().__init__()
        hidden_units = hidden_units or [256, 256]
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden in hidden_units:
            layers.extend([nn.Linear(current_dim, hidden), nn.SiLU()])
            current_dim = hidden
        self.backbone = nn.Sequential(*layers)
        self.value_head = nn.Linear(current_dim, 1)
        self.offtrack_head = nn.Linear(current_dim, 1)
        self.rank_head = nn.Linear(current_dim, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.backbone(features)
        return {
            "value_hat": self.value_head(hidden),
            "offtrack_prob": torch.sigmoid(self.offtrack_head(hidden)),
            "ranking_score": self.rank_head(hidden),
        }


class StudentIntentEncoder(nn.Module):
    def __init__(self, state_dim: int, residual_dim: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(state_dim, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU())
        self.latents = nn.Linear(256, residual_dim)
        self.plan_heads = nn.ModuleDict({field_name: nn.Linear(256, len(options)) for field_name, options in PLAN_CODE_SCHEMA.items()})
        self.confidence = nn.Linear(256, 1)
        self.offtrack = nn.Linear(256, 1)
        self.value = nn.Linear(256, 1)

    def forward(self, obs: torch.Tensor) -> dict[str, Any]:
        hidden = self.shared(obs)
        return {
            "z_mid": self.latents(hidden),
            "plan_logits": {field_name: head(hidden) for field_name, head in self.plan_heads.items()},
            "value_hat": self.value(hidden),
            "confidence": torch.sigmoid(self.confidence(hidden)),
            "offtrack_prob": torch.sigmoid(self.offtrack(hidden)),
        }


@dataclass
class RuntimeCache:
    output: UnifiedBackboneOutput
    step_index: int
    event_bits: list[float]
    narration: str = ""


class SharedBackboneRuntime:
    def __init__(self, *, state_dim: int, config: dict[str, Any], max_tasks: int, obs_channel_names: list[str] | None = None) -> None:
        self.config = dict(config)
        self.state_dim = int(state_dim)
        self.obs_channel_names = list(obs_channel_names or OBS_CHANNEL_NAMES_DEFAULT)
        self.backbone_device = _safe_torch_device(
            f"cuda:{int(self.config.get('backbone_device_index', 0))}" if torch.cuda.is_available() else "cpu"
        )
        self.encoder = UnifiedBackboneEncoder(
            state_dim=self.state_dim,
            model_name=str(self.config.get("model_name", "Qwen/Qwen3.5-4B-Base")),
            fallback_model_name=str(self.config.get("fallback_model_name", "Qwen/Qwen3.5-2B-Base")),
            device=self.backbone_device,
            max_tasks=max_tasks,
            summary_token_count=int(self.config.get("summary_token_count", 8)),
            quantization=str(self.config.get("quantization", "4bit_nf4")),
            branch_layer_4b=int(self.config.get("branch_layer_4b", 16)),
            branch_layer_2b=int(self.config.get("branch_layer_2b", 12)),
        )
        self.hidden_size = self.encoder.backbone.hidden_size
        self.plan_head = PlanHead(self.hidden_size).to(self.backbone_device)
        self.value_head = ValueHead(self.hidden_size).to(self.backbone_device)
        self.risk_head = RiskHead(self.hidden_size).to(self.backbone_device)
        self.student = StudentIntentEncoder(self.state_dim, self.hidden_size).to(self.backbone_device)
        self.frame_buffer_len = int(self.config.get("frame_buffer_len", 64))
        self.cache_refresh_hz = float(self.config.get("cache_refresh_hz", 5.0))
        self.ctrl_rate_hz = float(self.config.get("ctrl_rate_hz", 25.0))
        self.refresh_every_steps = max(1, int(round(self.ctrl_rate_hz / max(self.cache_refresh_hz, 1e-6))))
        self.stale_step_limit = int(self.config.get("stale_step_limit", 10))
        self._frame_buffer: deque[np.ndarray] = deque(maxlen=self.frame_buffer_len)
        self._event_buffer: deque[np.ndarray] = deque(maxlen=self.frame_buffer_len)
        self._task_buffer: deque[int] = deque(maxlen=self.frame_buffer_len)
        self._cache: RuntimeCache | None = None
        self._step_index = 0
        self._latency_window_ms: deque[float] = deque(maxlen=256)
        self._confidence_window: deque[float] = deque(maxlen=256)
        self._fallback_mode = str(self.config.get("fallback_mode", "student_then_zero"))

        # Async backbone refresh — decouple inference from control loop
        self._refresh_lock = threading.Lock()
        self._refresh_pending = threading.Event()
        self._shutdown = threading.Event()
        self._refresh_thread = threading.Thread(target=self._worker_loop, daemon=True, name="backbone-refresh")
        self._refresh_thread.start()

    def freeze_backbone(self) -> None:
        for module in (self.encoder, self.plan_head, self.value_head, self.risk_head, self.student):
            for param in module.parameters():
                param.requires_grad = False
            module.eval()

    def build_event_bits(self, env_info: dict[str, Any] | None) -> list[float]:
        if not env_info:
            return [0.0] * 6
        ub = env_info.get("unified_backbone", env_info)
        if "event_bits" in ub:
            values = [float(v) for v in ub["event_bits"]]
            if len(values) < 6:
                values.extend([0.0] * (6 - len(values)))
            return values[:6]
        return [
            float(bool(ub.get("corner_entry", False))),
            float(bool(ub.get("overspeed_event", False))),
            float(bool(ub.get("gap_alert", False))),
            float(bool(ub.get("heading_alert", False))),
            float(bool(ub.get("recovery_event", False))),
            float(bool(ub.get("off_track_recent", False))),
        ]

    def task_id_from_info(self, env_info: dict[str, Any] | None) -> int:
        if not env_info:
            return 0
        ub = env_info.get("unified_backbone", env_info)
        return int(ub.get("task_id", 0))

    def reset(self, initial_state: np.ndarray, env_info: dict[str, Any] | None = None) -> UnifiedBackboneOutput:
        self._frame_buffer.clear()
        self._event_buffer.clear()
        self._task_buffer.clear()
        self._cache = None
        self._step_index = 0
        self.observe(initial_state, env_info=env_info, force_refresh=True)
        return self.latest_output()

    def observe(self, state: np.ndarray, *, env_info: dict[str, Any] | None = None, force_refresh: bool = False) -> UnifiedBackboneOutput:
        state_array = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_array.shape[0] != self.state_dim:
            raise ValueError(f"Expected state_dim={self.state_dim}, got {state_array.shape[0]}")
        self._frame_buffer.append(state_array)
        self._event_buffer.append(np.asarray(self.build_event_bits(env_info), dtype=np.float32))
        self._task_buffer.append(self.task_id_from_info(env_info))
        self._step_index += 1
        if force_refresh:
            # Synchronous refresh only on episode reset (first observation)
            self._refresh_cache()
        elif self._should_refresh(env_info):
            # Non-blocking: signal background thread to refresh
            self._refresh_pending.set()
        return self.latest_output()

    def _should_refresh(self, env_info: dict[str, Any] | None) -> bool:
        if self._cache is None:
            return True
        if (self._step_index - self._cache.step_index) >= self.refresh_every_steps:
            return True
        if (self._step_index - self._cache.step_index) >= self.stale_step_limit:
            return True
        event_bits = self.build_event_bits(env_info)
        if any(bit > 0.5 for bit in event_bits[:5]):
            return True
        return self._cache.output.confidence < 0.4

    def _sequence_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_seq = np.stack(self._frame_buffer, axis=0)[None, ...]
        event_seq = np.stack(self._event_buffer, axis=0)[None, ...]
        if len(self._frame_buffer) > 1:
            previous = np.stack(list(self._frame_buffer)[:-1], axis=0)
            current = np.stack(list(self._frame_buffer)[1:], axis=0)
            delta = np.vstack([np.zeros((1, self.state_dim), dtype=np.float32), current - previous])[None, ...]
        else:
            delta = np.zeros_like(obs_seq)
        task_ids = np.asarray(list(self._task_buffer), dtype=np.int64)[None, ...]
        return (
            torch.tensor(obs_seq, dtype=torch.float32, device=self.backbone_device),
            torch.tensor(delta, dtype=torch.float32, device=self.backbone_device),
            torch.tensor(event_seq, dtype=torch.float32, device=self.backbone_device),
            torch.tensor(task_ids, dtype=torch.long, device=self.backbone_device),
        )

    def _refresh_cache(self) -> None:
        start = time.perf_counter()
        try:
            obs_seq, delta_seq, event_bits, task_ids = self._sequence_tensors()
            with torch.no_grad():
                encoded = self.encoder(obs_seq, delta_seq=delta_seq, task_ids=task_ids, event_bits=event_bits)
                z_mid = encoded["z_mid"]
                plan_logits = self.plan_head(z_mid)
                plan_ids = {field_name: int(torch.argmax(logits, dim=-1).item()) for field_name, logits in plan_logits.items()}
                confidence, offtrack_prob = self.risk_head(z_mid)
                output = UnifiedBackboneOutput(
                    z_mid=z_mid.squeeze(0).detach().cpu().float().tolist(),
                    plan_code=plan_code_from_ids(plan_ids),
                    plan_code_ids=plan_ids,
                    plan_logits={field_name: logits.squeeze(0).detach().cpu().float().tolist() for field_name, logits in plan_logits.items()},
                    value_hat=float(self.value_head(z_mid).item()),
                    confidence=float(confidence.item()),
                    offtrack_prob=float(offtrack_prob.item()),
                    planner_version=self.encoder.backbone.active_model_name,
                    valid=True,
                )
        except Exception:
            output = self._student_or_zero_fallback()

        output.latency_ms = float((time.perf_counter() - start) * 1000.0)
        self._latency_window_ms.append(output.latency_ms)
        self._confidence_window.append(output.confidence)

        # Build narration from the latest frame
        narration = ""
        if self._frame_buffer:
            narration = self._narrate_state(self._frame_buffer[-1], output)

        with self._refresh_lock:
            self._cache = RuntimeCache(
                output=output,
                step_index=self._step_index,
                event_bits=list(self._event_buffer[-1]) if self._event_buffer else [0.0] * 6,
                narration=narration,
            )

        if len(self._latency_window_ms) >= 32:
            latency_p95 = float(np.quantile(np.asarray(self._latency_window_ms, dtype=np.float32), 0.95))
            if latency_p95 > 120.0:
                logger.warning(
                    "Planner latency p95 %.2f ms exceeds threshold, but runtime backbone switching is disabled after init to keep hidden sizes stable.",
                    latency_p95,
                )

    def _worker_loop(self) -> None:
        """Background thread: waits for refresh signal, runs backbone inference."""
        while not self._shutdown.is_set():
            self._refresh_pending.wait(timeout=1.0)
            self._refresh_pending.clear()
            if self._shutdown.is_set():
                break
            try:
                self._refresh_cache()
            except Exception:
                logger.debug("Background backbone refresh failed", exc_info=True)

    def shutdown(self) -> None:
        """Stop the background refresh thread."""
        self._shutdown.set()
        self._refresh_pending.set()  # wake up worker so it can exit
        if self._refresh_thread.is_alive():
            self._refresh_thread.join(timeout=5.0)

    def _student_or_zero_fallback(self) -> UnifiedBackboneOutput:
        if self._fallback_mode != "student_then_zero" or not self._frame_buffer:
            return UnifiedBackboneOutput(plan_code=dict(DEFAULT_PLAN_CODE), planner_version="zero_fallback", valid=False)
        last_state = torch.tensor(self._frame_buffer[-1][None, :], dtype=torch.float32, device=self.backbone_device)
        with torch.no_grad():
            student = self.student(last_state)
        plan_ids = {field_name: int(torch.argmax(logits, dim=-1).item()) for field_name, logits in student["plan_logits"].items()}
        return UnifiedBackboneOutput(
            z_mid=student["z_mid"].squeeze(0).detach().cpu().float().tolist(),
            plan_code=plan_code_from_ids(plan_ids),
            plan_code_ids=plan_ids,
            plan_logits={field_name: logits.squeeze(0).detach().cpu().float().tolist() for field_name, logits in student["plan_logits"].items()},
            value_hat=float(student["value_hat"].item()),
            confidence=float(student["confidence"].item()),
            offtrack_prob=float(student["offtrack_prob"].item()),
            planner_version="student_fallback",
            valid=False,
        )

    def latest_output(self) -> UnifiedBackboneOutput:
        with self._refresh_lock:
            if self._cache is None:
                return UnifiedBackboneOutput(plan_code=dict(DEFAULT_PLAN_CODE), planner_version="empty_cache", valid=False)
            return self._cache.output

    def latest_narration(self) -> str:
        with self._refresh_lock:
            if self._cache is None:
                return ""
            return self._cache.narration

    def latest_control_context(self) -> dict[str, Any]:
        output = self.latest_output()
        return {
            "speed_mode": output.plan_code.get("speed_mode", "nominal"),
            "recovery_mode": output.plan_code.get("recovery_mode", "off"),
            "risk_mode": output.plan_code.get("risk_mode", "low"),
            "confidence": output.confidence,
            "value_hat": output.value_hat,
            "valid": output.valid,
            "planner_version": output.planner_version,
        }

    def encode_batch_states(
        self,
        states: torch.Tensor,
        *,
        task_ids: torch.Tensor | None = None,
        event_bits: torch.Tensor | None = None,
        require_top: bool = False,
        enable_grad: bool = False,
        prev_states: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if states.dim() != 2:
            raise ValueError("states must have shape [batch, state_dim]")
        obs_seq = states[:, None, :].to(self.backbone_device)
        if prev_states is not None:
            delta_seq = (states - prev_states)[:, None, :].to(self.backbone_device)
        else:
            delta_seq = torch.zeros_like(obs_seq)
        if event_bits is None:
            event_bits = torch.zeros(states.shape[0], 1, 6, dtype=torch.float32, device=self.backbone_device)
        elif event_bits.dim() == 2:
            event_bits = event_bits[:, None, :].to(self.backbone_device)
        else:
            event_bits = event_bits.to(self.backbone_device)
        if task_ids is None:
            task_ids = torch.zeros(states.shape[0], 1, dtype=torch.long, device=self.backbone_device)
        elif task_ids.dim() == 1:
            task_ids = task_ids[:, None].to(self.backbone_device)
        else:
            task_ids = task_ids.to(self.backbone_device)

        encoded = self.encoder(obs_seq, delta_seq=delta_seq, task_ids=task_ids, event_bits=event_bits, require_top=require_top, enable_grad=enable_grad)
        z_mid = encoded["z_mid"]
        plan_logits = self.plan_head(z_mid)
        confidence, offtrack_prob = self.risk_head(z_mid)
        value_hat = self.value_head(z_mid)
        return {
            "z_mid": z_mid,
            "top_hidden": encoded["top_hidden"],
            "plan_logits": plan_logits,
            "plan_ids": {field_name: torch.argmax(logits, dim=-1) for field_name, logits in plan_logits.items()},
            "value_hat": value_hat,
            "confidence": confidence,
            "offtrack_prob": offtrack_prob,
        }

    def runtime_metrics(self) -> dict[str, float | str]:
        latencies = np.asarray(self._latency_window_ms, dtype=np.float32)
        confidences = np.asarray(self._confidence_window, dtype=np.float32)
        return {
            "cache_refresh_count": float(len(self._latency_window_ms)),
            "planner_confidence_mean": float(confidences.mean()) if len(confidences) else 0.0,
            "planner_latency_p95_ms": float(np.quantile(latencies, 0.95)) if len(latencies) else 0.0,
            "planner_latency_mean_ms": float(latencies.mean()) if len(latencies) else 0.0,
            "planner_model_name": self.encoder.backbone.active_model_name,
        }

    def save(self, save_path: str) -> None:
        torch.save(
            {
                "state_tokenizer": self.encoder.state_tokenizer.state_dict(),
                "temporal_compressor": self.encoder.temporal_compressor.state_dict(),
                "plan_head": self.plan_head.state_dict(),
                "value_head": self.value_head.state_dict(),
                "risk_head": self.risk_head.state_dict(),
                "student": self.student.state_dict(),
                "config": self.config,
                "metadata": {
                    "planner_model_name": self.encoder.backbone.active_model_name,
                    "hidden_size": self.hidden_size,
                    "num_layers": self.encoder.backbone.num_layers,
                },
            },
            save_path,
        )

    @staticmethod
    def _compatible_state_dict(module: nn.Module, loaded_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        current_state = module.state_dict()
        return {
            key: value
            for key, value in loaded_state.items()
            if key in current_state and tuple(value.shape) == tuple(current_state[key].shape)
        }

    def load(self, load_path: str) -> None:
        state = torch.load(load_path, map_location=self.backbone_device)
        if "encoder" in state:
            self.encoder.load_state_dict(self._compatible_state_dict(self.encoder, state["encoder"]), strict=False)
        else:
            if "state_tokenizer" in state:
                self.encoder.state_tokenizer.load_state_dict(
                    self._compatible_state_dict(self.encoder.state_tokenizer, state["state_tokenizer"]),
                    strict=False,
                )
            if "temporal_compressor" in state:
                self.encoder.temporal_compressor.load_state_dict(
                    self._compatible_state_dict(self.encoder.temporal_compressor, state["temporal_compressor"]),
                    strict=False,
                )
        if "plan_head" in state:
            self.plan_head.load_state_dict(self._compatible_state_dict(self.plan_head, state["plan_head"]), strict=False)
        if "value_head" in state:
            self.value_head.load_state_dict(self._compatible_state_dict(self.value_head, state["value_head"]), strict=False)
        if "risk_head" in state:
            self.risk_head.load_state_dict(self._compatible_state_dict(self.risk_head, state["risk_head"]), strict=False)
        if "student" in state:
            self.student.load_state_dict(self._compatible_state_dict(self.student, state["student"]), strict=False)

    def _narrate_state(self, state: np.ndarray, output: UnifiedBackboneOutput) -> str:
        """Convert raw observation + backbone output into a human-readable narration."""
        s = state
        n = len(self.obs_channel_names)
        parts: list[str] = []

        # Extract key telemetry by channel name if available
        ch = {}
        if n > 0 and len(s) >= n:
            for i, name in enumerate(self.obs_channel_names[:len(s)]):
                ch[name] = float(s[i])
        elif len(s) >= 14:
            # Fallback: use positional indices for the 14 base channels
            _base = ["speed", "gap", "last_ff", "rpm", "accel_x", "accel_y",
                     "gear", "yaw_rate", "velocity_x", "velocity_y",
                     "slip_fl", "slip_fr", "slip_rl", "slip_rr"]
            for i, name in enumerate(_base):
                ch[name] = float(s[i])

        if not ch:
            return ""

        # Speed and gap
        speed_ms = ch.get("speed", 0.0)
        speed_kph = speed_ms * 3.6
        gap = ch.get("gap", 0.0)
        gear = int(round(ch.get("gear", 0.0)))
        parts.append(f"Speed {speed_kph:.0f} km/h (gear {gear}), gap {gap:+.2f}m from racing line.")

        # Curvature trend
        curv_vals = [ch.get(f"curvature_{i}", 0.0) for i in range(12)]
        max_curv = max(abs(c) for c in curv_vals) if curv_vals else 0.0
        if max_curv > 0.08:
            direction = "left" if sum(curv_vals[:6]) > 0 else "right"
            parts.append(f"Approaching {'tight' if max_curv > 0.15 else 'moderate'} {direction} turn (peak curvature {max_curv:.3f}).")
        else:
            parts.append("Straight or gentle curve ahead.")

        # Slip / stability
        slip_rl = abs(ch.get("slip_rl", 0.0))
        slip_rr = abs(ch.get("slip_rr", 0.0))
        rear_slip = max(slip_rl, slip_rr)
        if rear_slip > 2.0:
            parts.append(f"Rear slip angle {rear_slip:.1f} deg — {'heavy' if rear_slip > 5.0 else 'light'} oversteer.")

        # Plan code and predictions
        pc = output.plan_code
        parts.append(
            f"Plan: {pc.get('speed_mode','?')}/{pc.get('brake_phase','?')}/{pc.get('line_mode','?')}, "
            f"stability={pc.get('stability_mode','?')}, recovery={pc.get('recovery_mode','?')}. "
            f"Value={output.value_hat:.2f}, risk={pc.get('risk_mode','?')}, confidence={output.confidence:.2f}."
        )
        return " ".join(parts)

    def generate_coach_response(self, *, question: str, segment_summary: dict[str, Any], evidence_segments: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        evidence_segments = evidence_segments or []
        evidence_ids = [segment.get("segment_id", "") for segment in evidence_segments[:4] if segment.get("segment_id")]
        plan_code = canonicalize_plan_code(segment_summary.get("plan_code"))
        narration = self.latest_narration()
        narration_block = f"Current state: {narration}\n" if narration else ""
        prompt = (
            "You are a grounded racing coach. Answer in 2-3 sentences using only the supplied telemetry evidence.\n"
            f"{narration_block}"
            f"Question: {question}\nPlan code: {plan_code}\nEvidence ids: {evidence_ids}\nSummary: {segment_summary}\n"
        )
        generated = self.encoder.backbone.generate_text(prompt)
        if not generated:
            generated = (
                f"Plan state is speed={plan_code['speed_mode']}, brake={plan_code['brake_phase']}, "
                f"line={plan_code['line_mode']}, stability={plan_code['stability_mode']}, "
                f"recovery={plan_code['recovery_mode']}, risk={plan_code['risk_mode']}. "
                "Brake slightly earlier into the next corner entry and prioritize a cleaner exit."
            )
        return {
            "answer": generated.strip(),
            "plan_code": plan_code,
            "evidence_segment_ids": evidence_ids,
            "model_name": self.encoder.backbone.active_model_name,
        }
