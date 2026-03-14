from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from planner.schemas import PLAN_CODE_SCHEMA, json_loads
from planner.unified_backbone import SegmentValueNet, SharedBackboneRuntime


class SegmentDataset(Dataset):
    def __init__(self, segment_store: Path) -> None:
        self.frame = pd.read_parquet(segment_store)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[int(index)]
        frame_observations = json_loads(row["frame_observations_json"], default=[])
        frame_event_bits = json_loads(row["frame_event_bits_json"], default=[])
        raw_features = json_loads(row["raw_numeric_features_json"], default={})
        plan_code_ids = json_loads(row["plan_code_ids_json"], default={})
        return {
            "obs_seq": torch.tensor(frame_observations, dtype=torch.float32),
            "event_bits": torch.tensor(frame_event_bits, dtype=torch.float32),
            "task_id": torch.tensor(int(row["task_id"]), dtype=torch.long),
            "numeric_features": torch.tensor([float(raw_features[key]) for key in sorted(raw_features.keys())], dtype=torch.float32),
            "plan_code_ids": {
                field_name: torch.tensor(int(plan_code_ids.get(field_name, 0)), dtype=torch.long)
                for field_name in PLAN_CODE_SCHEMA.keys()
            },
            "future_progress_3s": torch.tensor(float(row["future_progress_3s"]), dtype=torch.float32),
            "offtrack_next_3s": torch.tensor(float(row["offtrack_next_3s"]), dtype=torch.float32),
        }


def collate_segments(batch: list[dict]) -> dict:
    max_steps = max(item["obs_seq"].shape[0] for item in batch)
    state_dim = batch[0]["obs_seq"].shape[-1]
    event_dim = batch[0]["event_bits"].shape[-1]
    obs = torch.zeros(len(batch), max_steps, state_dim, dtype=torch.float32)
    events = torch.zeros(len(batch), max_steps, event_dim, dtype=torch.float32)
    task_ids = torch.zeros(len(batch), max_steps, dtype=torch.long)
    numeric_features = torch.stack([item["numeric_features"] for item in batch], dim=0)
    plan_code_ids = {field_name: torch.stack([item["plan_code_ids"][field_name] for item in batch], dim=0) for field_name in PLAN_CODE_SCHEMA.keys()}
    future_progress = torch.stack([item["future_progress_3s"] for item in batch], dim=0).unsqueeze(-1)
    offtrack = torch.stack([item["offtrack_next_3s"] for item in batch], dim=0).unsqueeze(-1)

    for index, item in enumerate(batch):
        steps = item["obs_seq"].shape[0]
        obs[index, :steps] = item["obs_seq"]
        events[index, :steps] = item["event_bits"]
        task_ids[index, :steps] = item["task_id"]
    return {
        "obs_seq": obs,
        "event_bits": events,
        "task_ids": task_ids,
        "numeric_features": numeric_features,
        "plan_code_ids": plan_code_ids,
        "future_progress_3s": future_progress,
        "offtrack_next_3s": offtrack,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline pretrain the unified backbone heads from segment parquet data.")
    parser.add_argument("--segment-store", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B-Base")
    parser.add_argument("--fallback-model-name", default="Qwen/Qwen3.5-2B-Base")
    parser.add_argument("--quantization", default="4bit_nf4")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = SegmentDataset(Path(args.segment_store))
    if len(dataset) == 0:
        raise RuntimeError("Segment dataset is empty.")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_segments)

    state_dim = dataset[0]["obs_seq"].shape[-1]
    feature_dim = dataset[0]["numeric_features"].shape[-1]
    runtime = SharedBackboneRuntime(
        state_dim=state_dim,
        config={
            "model_name": args.model_name,
            "fallback_model_name": args.fallback_model_name,
            "quantization": args.quantization,
            "backbone_device_index": args.cuda_device,
            "summary_token_count": 8,
            "frame_buffer_len": 64,
            "cache_refresh_hz": 5.0,
            "branch_layer_4b": 16,
            "branch_layer_2b": 12,
        },
        max_tasks=int(dataset.frame["task_id"].max()) + 1,
    )
    for param in runtime.encoder.backbone.parameters():
        param.requires_grad = False
    runtime.encoder.backbone.eval()
    segment_value_net = SegmentValueNet(input_dim=runtime.hidden_size + feature_dim).to(runtime.backbone_device)

    trainable_modules = [
        runtime.encoder.state_tokenizer,
        runtime.encoder.temporal_compressor,
        runtime.plan_head,
        runtime.value_head,
        runtime.risk_head,
        segment_value_net,
    ]
    parameters = []
    for module in trainable_modules:
        module.train()
        parameters.extend(list(module.parameters()))

    optimizer = AdamW(parameters, lr=args.lr)
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    metrics = []

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        steps = 0
        for batch in loader:
            obs_seq = batch["obs_seq"].to(runtime.backbone_device)
            event_bits = batch["event_bits"].to(runtime.backbone_device)
            task_ids = batch["task_ids"].to(runtime.backbone_device)
            numeric_features = batch["numeric_features"].to(runtime.backbone_device)
            future_progress = batch["future_progress_3s"].to(runtime.backbone_device)
            offtrack = batch["offtrack_next_3s"].to(runtime.backbone_device)

            encoded = runtime.encoder(obs_seq, event_bits=event_bits, task_ids=task_ids)
            z_mid = encoded["z_mid"]
            plan_logits = runtime.plan_head(z_mid)
            value_hat = runtime.value_head(z_mid)
            confidence, offtrack_prob = runtime.risk_head(z_mid)
            value_outputs = segment_value_net(torch.cat([z_mid, numeric_features], dim=-1))

            loss = mse_loss(value_hat, future_progress)
            loss = loss + mse_loss(value_outputs["value_hat"], future_progress)
            loss = loss + bce_loss(offtrack_prob, offtrack)
            loss = loss + bce_loss(value_outputs["offtrack_prob"], offtrack)
            loss = loss + mse_loss(confidence, 1.0 - offtrack)
            for field_name, logits in plan_logits.items():
                loss = loss + ce_loss(logits, batch["plan_code_ids"][field_name].to(runtime.backbone_device))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            steps += 1

        epoch_metric = {"epoch": epoch + 1, "loss": epoch_loss / max(steps, 1)}
        metrics.append(epoch_metric)
        print(epoch_metric)

    runtime.save(str(output_dir / "shared_backbone_bundle.pth"))
    torch.save(segment_value_net.state_dict(), output_dir / "segment_value_net.pth")
    (output_dir / "offline_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
