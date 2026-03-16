import os
from contextlib import nullcontext
import logging

import torch
from torch import nn
from torch.distributions import Normal
from torch.optim import Adam

from .base import Algorithm
from discor.utils import assert_action, disable_gradients, soft_update, update_params
from planner.schemas import PLAN_CODE_SCHEMA
from planner.unified_backbone import SharedBackboneRuntime

logger = logging.getLogger(__name__)


class FiLMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, cond_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.scale = nn.Linear(cond_dim, output_dim)
        self.shift = nn.Linear(cond_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x, cond):
        base = self.linear(x)
        scale = 1.0 + self.scale(cond)
        shift = self.shift(cond)
        return self.norm(base * scale + shift)


class FiLMBlock(nn.Module):
    def __init__(self, input_dim, cond_dim, hidden_units):
        super().__init__()
        self.layers = nn.ModuleList()
        current_dim = input_dim
        for hidden_dim in hidden_units:
            self.layers.append(FiLMLayer(current_dim, hidden_dim, cond_dim))
            current_dim = hidden_dim
        self.output_dim = current_dim

    def forward(self, x, cond):
        for layer in self.layers:
            x = torch.nn.functional.silu(layer(x, cond))
        return x


class FiLMGaussianPolicy(nn.Module):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20

    def __init__(self, input_dim, cond_dim, action_dim, hidden_units, mean_bias=None, log_std_bias=None):
        super().__init__()
        self.action_dim = action_dim
        self.backbone = FiLMBlock(input_dim, cond_dim, hidden_units)
        self.output = nn.Linear(self.backbone.output_dim, 2 * action_dim)
        self._apply_output_bias(mean_bias, log_std_bias)

    def _apply_output_bias(self, mean_bias, log_std_bias):
        if self.output.bias is None:
            return
        with torch.no_grad():
            if mean_bias is not None:
                mean_bias_tensor = torch.as_tensor(mean_bias, dtype=self.output.bias.dtype)
                self.output.bias[: self.action_dim].copy_(mean_bias_tensor)
            if log_std_bias is not None:
                log_std_tensor = torch.as_tensor(log_std_bias, dtype=self.output.bias.dtype)
                self.output.bias[self.action_dim :].copy_(log_std_tensor)

    def forward(self, x, cond):
        hidden = self.backbone(x, cond)
        means, log_stds = torch.chunk(self.output(hidden), 2, dim=-1)
        log_stds = torch.clamp(log_stds, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)
        normals = Normal(means, log_stds.exp())
        xs = normals.rsample()
        actions = torch.tanh(xs)
        log_probs = normals.log_prob(xs) - torch.log(1 - actions.pow(2) + 1e-6)
        entropies = -log_probs.sum(dim=1, keepdim=True)
        return actions, entropies, torch.tanh(means)


class FiLMTwinnedStateActionFunction(nn.Module):
    def __init__(self, state_dim, cond_dim, action_dim, hidden_units):
        super().__init__()
        self.q1 = FiLMBlock(state_dim + action_dim, cond_dim, hidden_units)
        self.q2 = FiLMBlock(state_dim + action_dim, cond_dim, hidden_units)
        self.q1_out = nn.Linear(self.q1.output_dim, 1)
        self.q2_out = nn.Linear(self.q2.output_dim, 1)

    def forward(self, states, actions, cond):
        x = torch.cat([states, actions], dim=-1)
        return self.q1_out(self.q1(x, cond)), self.q2_out(self.q2(x, cond))


class PlanConditionEncoder(nn.Module):
    def __init__(self, cond_dim, hidden_size, plan_schema):
        super().__init__()
        per_field_dim = max(8, cond_dim // max(1, len(plan_schema)))
        self.per_field_dim = per_field_dim
        self.embeddings = nn.ModuleDict({
            field_name: nn.Embedding(len(options), per_field_dim)
            for field_name, options in plan_schema.items()
        })
        numeric_dim = hidden_size + 4
        self.numeric = nn.Sequential(
            nn.Linear(numeric_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.cond_dim = cond_dim + len(plan_schema) * per_field_dim

    def forward(self, plan_ids, z_mid, value_hat, confidence, offtrack_prob, valid):
        parts = []
        for field_name, embedding in self.embeddings.items():
            parts.append(embedding(plan_ids[field_name].long()))
        parts.append(
            self.numeric(
                torch.cat(
                    [
                        z_mid,
                        value_hat,
                        confidence,
                        offtrack_prob,
                        valid,
                    ],
                    dim=-1,
                )
            )
        )
        return torch.cat(parts, dim=-1)


class SharedBackboneSAC(Algorithm):
    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        gamma=0.99,
        nstep=1,
        policy_lr=0.0003,
        q_lr=0.0003,
        entropy_lr=0.0003,
        policy_hidden_units=[256, 256, 256],
        q_hidden_units=[256, 256, 256],
        target_update_coef=0.005,
        log_interval=10,
        seed=0,
        policy_mean_bias=None,
        policy_log_std_bias=None,
        unified_backbone_config=None,
        obs_channel_names=None,
        task_id_dim=1,
        control_delta_dim=128,
    ):
        super().__init__(state_dim, action_dim, device, gamma, nstep, log_interval, seed)
        self._head_device = device
        self._cfg = dict(unified_backbone_config or {})
        self._freeze_backbone = bool(self._cfg.get("freeze_backbone", True))
        self._runtime = SharedBackboneRuntime(
            state_dim=self._state_dim,
            config=self._cfg,
            max_tasks=max(1, int(task_id_dim)),
            obs_channel_names=obs_channel_names,
        )
        if self._freeze_backbone:
            self._runtime.freeze_backbone()

        self._z_dim = self._runtime.hidden_size
        self._obs_projection = nn.Sequential(
            nn.Linear(self._state_dim, control_delta_dim),
            nn.SiLU(),
            nn.Linear(control_delta_dim, control_delta_dim),
        ).to(self._head_device)
        self._delta_projection = nn.Sequential(
            nn.Linear(self._state_dim, control_delta_dim),
            nn.SiLU(),
            nn.Linear(control_delta_dim, control_delta_dim),
        ).to(self._head_device)
        self._core_dim = self._z_dim + 2 * control_delta_dim
        self._cond_encoder = PlanConditionEncoder(
            cond_dim=128,
            hidden_size=self._z_dim,
            plan_schema=PLAN_CODE_SCHEMA,
        ).to(self._head_device)
        self._cond_dim = self._cond_encoder.cond_dim

        self._policy_net = FiLMGaussianPolicy(
            input_dim=self._core_dim,
            cond_dim=self._cond_dim,
            action_dim=self._action_dim,
            hidden_units=policy_hidden_units,
            mean_bias=policy_mean_bias,
            log_std_bias=policy_log_std_bias,
        ).to(self._head_device)
        self._online_q_net = FiLMTwinnedStateActionFunction(
            state_dim=self._core_dim,
            cond_dim=self._cond_dim,
            action_dim=self._action_dim,
            hidden_units=q_hidden_units,
        ).to(self._head_device)
        self._target_q_net = FiLMTwinnedStateActionFunction(
            state_dim=self._core_dim,
            cond_dim=self._cond_dim,
            action_dim=self._action_dim,
            hidden_units=q_hidden_units,
        ).to(self._head_device).eval()
        self._target_q_net.load_state_dict(self._online_q_net.state_dict())
        disable_gradients(self._target_q_net)

        self._policy_optim = Adam(list(self._policy_net.parameters()), lr=policy_lr)
        self._q_optim = Adam(
            list(self._online_q_net.parameters())
            + list(self._obs_projection.parameters())
            + list(self._delta_projection.parameters())
            + list(self._cond_encoder.parameters()),
            lr=q_lr,
        )
        self._log_alpha = torch.zeros(1, device=self._head_device, requires_grad=True)
        self._alpha = self._log_alpha.detach().exp()
        self._alpha_optim = Adam([self._log_alpha], lr=entropy_lr)
        self._target_entropy = -float(self._action_dim)
        self._target_update_coef = target_update_coef
        self.update_entropy = True
        self._previous_state = None
        self._last_control_context = {}

        # Backbone training config
        self._backbone_update_every = int(self._cfg.get("backbone_update_every", 100))
        self._student_distill_weight = float(self._cfg.get("student_distill_weight", 0.5))
        self._backbone_optim = None
        self._backbone_ce = nn.CrossEntropyLoss()
        self._backbone_mse = nn.MSELoss()
        self._backbone_bce = nn.BCELoss()

        # Set up LoRA if configured
        lora_enabled = bool(self._cfg.get("lora_enabled", False))
        if lora_enabled and not self._freeze_backbone:
            lora_rank = int(self._cfg.get("lora_rank", 16))
            lora_alpha = int(self._cfg.get("lora_alpha", 32))
            if self._runtime.encoder.backbone.enable_lora(rank=lora_rank, alpha=lora_alpha):
                if bool(self._cfg.get("enable_grad_checkpointing", False)):
                    self._runtime.encoder.backbone.enable_gradient_checkpointing()
                self._setup_backbone_optimizer()

    def _setup_backbone_optimizer(self):
        """Create separate optimizer for backbone (LoRA + heads)."""
        lora_lr = float(self._cfg.get("lora_lr", 1e-5))
        head_lr = float(self._cfg.get("head_lr", 3e-4))

        param_groups = []
        # LoRA parameters (low LR)
        lora_params = self._runtime.encoder.backbone.lora_parameters()
        if lora_params:
            param_groups.append({"params": lora_params, "lr": lora_lr})

        # Head parameters (higher LR)
        head_modules = [
            self._runtime.encoder.state_tokenizer,
            self._runtime.encoder.temporal_compressor,
            self._runtime.plan_head,
            self._runtime.value_head,
            self._runtime.risk_head,
            self._runtime.student,
        ]
        head_params = []
        for module in head_modules:
            head_params.extend([p for p in module.parameters() if p.requires_grad])
        if head_params:
            param_groups.append({"params": head_params, "lr": head_lr})

        if param_groups:
            self._backbone_optim = Adam(param_groups)
            logger.info(
                "Backbone optimizer created: %d LoRA params (lr=%.1e), %d head params (lr=%.1e)",
                len(lora_params), lora_lr, len(head_params), head_lr,
            )

    def reset_runtime_context(self, initial_state, env_info=None):
        self._previous_state = None
        self._runtime.reset(initial_state, env_info=env_info)
        self._last_control_context = self._runtime.latest_control_context()

    def observe_env_info(self, info, next_state=None, reward=None, done=None):
        if next_state is None:
            return
        self._runtime.observe(next_state, env_info=info)
        self._last_control_context = self._runtime.latest_control_context()
        self._previous_state = next_state

    def get_latest_control_context(self):
        return dict(self._last_control_context)

    def get_runtime_metrics(self):
        return dict(self._runtime.runtime_metrics())

    def _load_module_if_compatible(self, module, path):
        if not os.path.exists(path):
            return
        loaded_state = torch.load(path, map_location=self._head_device)
        current_state = module.state_dict()
        compatible = {
            key: value
            for key, value in loaded_state.items()
            if key in current_state and tuple(value.shape) == tuple(current_state[key].shape)
        }
        module.load_state_dict(compatible, strict=False)

    def _live_features(self, state):
        if self._runtime.latest_output().planner_version == "empty_cache":
            self.reset_runtime_context(state)

        output = self._runtime.latest_output()
        if output.z_mid:
            z_mid = torch.tensor(output.z_mid, dtype=torch.float32, device=self._head_device).unsqueeze(0)
        else:
            z_mid = torch.zeros(1, self._z_dim, dtype=torch.float32, device=self._head_device)

        state_tensor = torch.tensor(state[None, ...].copy(), dtype=torch.float32, device=self._head_device)
        if self._previous_state is None:
            delta = torch.zeros_like(state_tensor)
        else:
            delta = state_tensor - torch.tensor(self._previous_state[None, ...].copy(), dtype=torch.float32, device=self._head_device)
        core = torch.cat([z_mid, self._obs_projection(state_tensor), self._delta_projection(delta)], dim=-1)

        plan_ids = {
            field_name: torch.tensor([output.plan_code_ids.get(field_name, 0)], dtype=torch.long, device=self._head_device)
            for field_name in PLAN_CODE_SCHEMA.keys()
        }
        cond = self._cond_encoder(
            plan_ids=plan_ids,
            z_mid=z_mid,
            value_hat=torch.tensor([[output.value_hat]], dtype=torch.float32, device=self._head_device),
            confidence=torch.tensor([[output.confidence]], dtype=torch.float32, device=self._head_device),
            offtrack_prob=torch.tensor([[output.offtrack_prob]], dtype=torch.float32, device=self._head_device),
            valid=torch.tensor([[1.0 if output.valid else 0.0]], dtype=torch.float32, device=self._head_device),
        )
        return core, cond

    def _batch_features(self, states, event_bits=None, prev_states=None, enable_grad=False):
        grad_context = nullcontext() if (not self._freeze_backbone or enable_grad) else torch.no_grad()
        with grad_context:
            encoded = self._runtime.encode_batch_states(states, event_bits=event_bits, enable_grad=enable_grad, prev_states=prev_states)
        z_mid = encoded["z_mid"].to(self._head_device)
        state_tensor = states.to(self._head_device)
        if prev_states is not None:
            delta_tensor = state_tensor - prev_states.to(self._head_device)
        else:
            delta_tensor = torch.zeros_like(state_tensor)
        core = torch.cat([z_mid, self._obs_projection(state_tensor), self._delta_projection(delta_tensor)], dim=-1)
        plan_ids = {field_name: tensor.to(self._head_device) for field_name, tensor in encoded["plan_ids"].items()}
        cond = self._cond_encoder(
            plan_ids=plan_ids,
            z_mid=z_mid,
            value_hat=encoded["value_hat"].to(self._head_device),
            confidence=encoded["confidence"].to(self._head_device),
            offtrack_prob=encoded["offtrack_prob"].to(self._head_device),
            valid=torch.ones(states.shape[0], 1, dtype=torch.float32, device=self._head_device),
        )
        return core, cond

    def explore(self, state):
        core, cond = self._live_features(state)
        with torch.no_grad():
            action, entropies, _ = self._policy_net(core, cond)
        action = action.cpu().numpy()[0]
        assert_action(action)
        return action, entropies

    def exploit(self, state):
        core, cond = self._live_features(state)
        with torch.no_grad():
            _, entropies, action = self._policy_net(core, cond)
        action = action.cpu().numpy()[0]
        assert_action(action)
        return action, entropies

    def update_target_networks(self):
        soft_update(self._target_q_net, self._online_q_net, self._target_update_coef)

    def update_online_networks(self, batch, writer, *, source: str = "live"):
        self._learning_steps += 1
        stats = self.update_policy_and_entropy(batch, writer)
        self.update_q_functions(batch, writer)
        if hasattr(self, '_backbone_update_every') and self._learning_steps % self._backbone_update_every == 0:
            if source == "post_episode":
                # Only run expensive backbone gradient updates between episodes
                self.update_backbone(batch, writer)
            else:
                # Defer backbone update — it will run in post-episode replay burst
                self._backbone_update_pending = True
        return stats

    def update_policy_and_entropy(self, batch, writer):
        states, actions, rewards, next_states, dones = batch[:5]
        prev_states = batch[5] if len(batch) > 5 else None
        core, cond = self._batch_features(states, prev_states=prev_states)
        sampled_actions, entropies, _ = self._policy_net(core, cond)
        qs1, qs2 = self._online_q_net(core, sampled_actions, cond)
        qs = torch.min(qs1, qs2)
        policy_loss = torch.mean((-qs - self._alpha * entropies))
        update_params(self._policy_optim, policy_loss)

        entropy_loss = 0.0
        if self.update_entropy:
            entropy_loss = self.calc_entropy_loss(entropies.detach())
            update_params(self._alpha_optim, entropy_loss)
            entropy_loss = entropy_loss.detach().item()
        self._alpha = self._log_alpha.detach().exp()

        if self._learning_steps % self._log_interval == 0:
            writer.add_scalar("loss/policy", policy_loss.detach().item(), self._learning_steps)
            writer.add_scalar("loss/entropy", entropy_loss, self._learning_steps)
            writer.add_scalar("stats/alpha", self._alpha.item(), self._learning_steps)
            writer.add_scalar("stats/entropy", entropies.detach().mean().item(), self._learning_steps)
            return {
                "policy_loss": policy_loss.detach().item(),
                "entropy_loss": entropy_loss,
                "alpha": self._alpha.item(),
                "entropy": entropies.detach().mean().item(),
            }

    def calc_entropy_loss(self, entropies):
        return -torch.mean(self._log_alpha * (self._target_entropy - entropies))

    def update_q_functions(self, batch, writer):
        states, actions, rewards, next_states, dones = batch[:5]
        prev_states = batch[5] if len(batch) > 5 else None
        curr_core, curr_cond = self._batch_features(states, prev_states=prev_states)
        next_core, next_cond = self._batch_features(next_states, prev_states=states)
        curr_qs1, curr_qs2 = self._online_q_net(curr_core, actions.to(self._head_device), curr_cond)
        with torch.no_grad():
            next_actions, next_entropies, _ = self._policy_net(next_core, next_cond)
            next_qs1, next_qs2 = self._target_q_net(next_core, next_actions, next_cond)
            next_qs = torch.min(next_qs1, next_qs2) + self._alpha * next_entropies
            target_qs = rewards.to(self._head_device) + (1.0 - dones.to(self._head_device)) * self._discount * next_qs

        q1_loss = torch.mean((curr_qs1 - target_qs).pow(2))
        q2_loss = torch.mean((curr_qs2 - target_qs).pow(2))
        q_loss = q1_loss + q2_loss
        update_params(self._q_optim, q_loss)

        if self._learning_steps % self._log_interval == 0:
            writer.add_scalar("loss/Q", q_loss.detach().item(), self._learning_steps)
            writer.add_scalar("stats/mean_Q1", curr_qs1.detach().mean().item(), self._learning_steps)
            writer.add_scalar("stats/mean_Q2", curr_qs2.detach().mean().item(), self._learning_steps)
        return curr_qs1.detach(), curr_qs2.detach(), target_qs

    def update_backbone(self, batch, writer):
        """Update backbone LoRA + heads with hindsight labels and student distillation."""
        if self._backbone_optim is None:
            return

        states = batch[0]
        prev_states = batch[5] if len(batch) > 5 else None

        # Forward with gradients through LoRA
        encoded = self._runtime.encode_batch_states(
            states, enable_grad=True, prev_states=prev_states,
        )
        z_mid = encoded["z_mid"]
        plan_logits = encoded["plan_logits"]
        value_hat = encoded["value_hat"]
        offtrack_prob = encoded["offtrack_prob"]

        # Use actual returns from batch as value target (rewards are n-step discounted)
        rewards = batch[2].to(z_mid.device)
        value_loss = self._backbone_mse(value_hat, rewards)

        # Use done signal as a proxy for offtrack (terminated episodes often = offtrack)
        dones = batch[4].to(z_mid.device)
        risk_loss = self._backbone_bce(offtrack_prob, dones)

        # Plan head loss against hindsight labels if available
        # For now, use the model's own argmax predictions as soft targets
        # (will be replaced by hindsight labels from agent.py post-episode hook)
        plan_loss = torch.tensor(0.0, device=z_mid.device)
        if hasattr(self, '_cached_hindsight_targets') and self._cached_hindsight_targets is not None:
            for i, field_name in enumerate(PLAN_CODE_SCHEMA.keys()):
                if field_name in plan_logits:
                    targets = self._cached_hindsight_targets[:, i].to(z_mid.device)
                    plan_loss = plan_loss + self._backbone_ce(plan_logits[field_name], targets)

        # Student distillation loss
        student_out = self._runtime.student(states.to(self._runtime.backbone_device))
        distill_loss = self._backbone_mse(student_out["z_mid"], z_mid.detach())

        total_loss = value_loss + risk_loss + plan_loss + self._student_distill_weight * distill_loss

        self._backbone_optim.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for group in self._backbone_optim.param_groups for p in group["params"]],
            max_norm=1.0,
        )
        self._backbone_optim.step()

        if self._learning_steps % (self._log_interval * 10) == 0:
            writer.add_scalar("backbone/value_mse", value_loss.detach().item(), self._learning_steps)
            writer.add_scalar("backbone/risk_bce", risk_loss.detach().item(), self._learning_steps)
            writer.add_scalar("backbone/plan_ce", plan_loss.detach().item(), self._learning_steps)
            writer.add_scalar("backbone/student_distill", distill_loss.detach().item(), self._learning_steps)
            writer.add_scalar("backbone/total_loss", total_loss.detach().item(), self._learning_steps)

    def set_hindsight_targets(self, targets: torch.Tensor):
        """Cache hindsight plan code targets from post-episode labeling."""
        self._cached_hindsight_targets = targets

    def save_models(self, save_dir):
        super().save_models(save_dir)
        self._runtime.save(os.path.join(save_dir, "shared_backbone_bundle.pth"))
        torch.save(self._obs_projection.state_dict(), os.path.join(save_dir, "shared_obs_projection.pth"))
        torch.save(self._delta_projection.state_dict(), os.path.join(save_dir, "shared_delta_projection.pth"))
        torch.save(self._cond_encoder.state_dict(), os.path.join(save_dir, "shared_cond_encoder.pth"))
        torch.save(self._policy_net.state_dict(), os.path.join(save_dir, "policy_net.pth"))
        torch.save(self._online_q_net.state_dict(), os.path.join(save_dir, "online_q_net.pth"))
        torch.save(self._target_q_net.state_dict(), os.path.join(save_dir, "target_q_net.pth"))
        # Save LoRA adapters if enabled
        if self._runtime.encoder.backbone._lora_enabled:
            lora_dir = os.path.join(save_dir, "lora_adapters")
            os.makedirs(lora_dir, exist_ok=True)
            self._runtime.encoder.backbone.save_lora(lora_dir)

    def load_models(self, load_dir):
        shared_bundle = os.path.join(load_dir, "shared_backbone_bundle.pth")
        if os.path.exists(shared_bundle):
            self._runtime.load(shared_bundle)
        obs_projection_path = os.path.join(load_dir, "shared_obs_projection.pth")
        if os.path.exists(obs_projection_path):
            self._obs_projection.load_state_dict(torch.load(obs_projection_path, map_location=self._head_device))
        delta_projection_path = os.path.join(load_dir, "shared_delta_projection.pth")
        if os.path.exists(delta_projection_path):
            self._delta_projection.load_state_dict(torch.load(delta_projection_path, map_location=self._head_device))
        cond_encoder_path = os.path.join(load_dir, "shared_cond_encoder.pth")
        if os.path.exists(cond_encoder_path):
            self._cond_encoder.load_state_dict(torch.load(cond_encoder_path, map_location=self._head_device))
        self._load_module_if_compatible(self._policy_net, os.path.join(load_dir, "policy_net.pth"))
        self._load_module_if_compatible(self._online_q_net, os.path.join(load_dir, "online_q_net.pth"))
        self._load_module_if_compatible(self._target_q_net, os.path.join(load_dir, "target_q_net.pth"))
        # Load LoRA adapters if available
        lora_dir = os.path.join(load_dir, "lora_adapters")
        if os.path.exists(lora_dir):
            self._runtime.encoder.backbone.load_lora(lora_dir)
            if self._backbone_optim is None:
                self._setup_backbone_optimizer()
