from __future__ import annotations

import argparse
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn

from coach.service import CoachService, CoachServiceConfig
from planner.unified_backbone import SharedBackboneRuntime


class AnalyzeLapRequest(BaseModel):
    run_id: str
    episode: int


class CompareSegmentsRequest(BaseModel):
    segment_a_id: str
    segment_b_id: str


class ChatRequest(BaseModel):
    question: str
    segment_ids: list[str] = Field(default_factory=list)


def build_app(segment_store: Path, interaction_log: Path) -> FastAPI:
    runtime = SharedBackboneRuntime(
        state_dim=14,
        config={
            "model_name": "Qwen/Qwen3.5-4B-Base",
            "fallback_model_name": "Qwen/Qwen3.5-2B-Base",
            "quantization": "4bit_nf4",
            "backbone_device_index": 0,
            "summary_token_count": 8,
            "frame_buffer_len": 64,
            "cache_refresh_hz": 5.0,
            "branch_layer_4b": 16,
            "branch_layer_2b": 12,
        },
        max_tasks=8,
    )
    runtime.freeze_backbone()
    service = CoachService(
        runtime=runtime,
        config=CoachServiceConfig(segment_store=segment_store, interaction_log=interaction_log),
    )

    app = FastAPI(title="Unified Backbone Coach API", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/coach/analyze_lap")
    def analyze_lap(request: AnalyzeLapRequest) -> dict:
        return service.analyze_lap(run_id=request.run_id, episode=request.episode)

    @app.post("/coach/compare_segments")
    def compare_segments(request: CompareSegmentsRequest) -> dict:
        return service.compare_segments(request.segment_a_id, request.segment_b_id)

    @app.post("/coach/chat")
    def chat(request: ChatRequest) -> dict:
        return service.chat(request.question, segment_ids=request.segment_ids)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the grounded unified-backbone coach API.")
    parser.add_argument("--segment-store", required=True)
    parser.add_argument("--interaction-log", required=True)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = build_app(segment_store=Path(args.segment_store), interaction_log=Path(args.interaction_log))
    uvicorn.run(app, host=args.bind, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
