from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import uuid

import numpy as np
import pandas as pd

from planner.schemas import CoachInteractionRecord, json_loads
from planner.unified_backbone import SharedBackboneRuntime


@dataclass
class CoachServiceConfig:
    segment_store: Path
    interaction_log: Path
    top_k_neighbors: int = 6


class CoachService:
    def __init__(self, runtime: SharedBackboneRuntime, config: CoachServiceConfig) -> None:
        self.runtime = runtime
        self.config = config
        self.config.interaction_log.parent.mkdir(parents=True, exist_ok=True)
        self._segments = self._load_segments()
        self._feature_matrix = self._build_feature_matrix(self._segments)

    def _load_segments(self) -> pd.DataFrame:
        if not self.config.segment_store.exists():
            return pd.DataFrame()
        return pd.read_parquet(self.config.segment_store)

    def _build_feature_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        if frame.empty:
            return np.zeros((0, 1), dtype=np.float32)
        features = []
        for payload in frame["raw_numeric_features_json"].tolist():
            parsed = json_loads(payload, default={})
            ordered = [float(parsed[key]) for key in sorted(parsed.keys())]
            features.append(ordered)
        return np.asarray(features, dtype=np.float32)

    def _segment_row(self, segment_id: str) -> dict[str, Any] | None:
        if self._segments.empty:
            return None
        matches = self._segments.loc[self._segments["segment_id"] == segment_id]
        if matches.empty:
            return None
        return matches.iloc[0].to_dict()

    def nearest_neighbors(self, segment_id: str) -> list[dict[str, Any]]:
        if self._segments.empty or self._feature_matrix.shape[0] == 0:
            return []
        base_row = self._segment_row(segment_id)
        if base_row is None:
            return []
        base_index = int(self._segments.index[self._segments["segment_id"] == segment_id][0])
        base_vector = self._feature_matrix[base_index]
        distances = np.linalg.norm(self._feature_matrix - base_vector[None, :], axis=1)
        rows = []
        for idx in np.argsort(distances):
            if idx == base_index:
                continue
            row = self._segments.iloc[int(idx)].to_dict()
            row["distance"] = float(distances[idx])
            rows.append(row)
            if len(rows) >= self.config.top_k_neighbors:
                break
        return rows

    def analyze_lap(self, run_id: str, episode: int) -> dict[str, Any]:
        if self._segments.empty:
            return {"answer": "No segment store is available yet.", "evidence_segment_ids": []}
        episode_segments = self._segments.loc[
            (self._segments["run_id"] == run_id) & (self._segments["episode"] == int(episode))
        ].copy()
        if episode_segments.empty:
            return {"answer": f"No segments found for run={run_id} episode={episode}.", "evidence_segment_ids": []}
        episode_segments["future_progress_3s"] = episode_segments["future_progress_3s"].astype(float)
        focus_row = episode_segments.sort_values("future_progress_3s", ascending=True).iloc[0].to_dict()
        return self._respond(question=f"Analyze run {run_id} episode {episode}.", focus_segment=focus_row)

    def compare_segments(self, segment_a_id: str, segment_b_id: str) -> dict[str, Any]:
        segment_a = self._segment_row(segment_a_id)
        segment_b = self._segment_row(segment_b_id)
        if segment_a is None or segment_b is None:
            return {"answer": "One or both segments were not found in the segment store.", "evidence_segment_ids": []}
        return self._respond(
            question=f"Compare segment {segment_a_id} against {segment_b_id}.",
            focus_segment=segment_a,
            extra_segments=[segment_b],
        )

    def chat(self, question: str, segment_ids: list[str] | None = None) -> dict[str, Any]:
        segment_ids = segment_ids or []
        focus_segment = self._segment_row(segment_ids[0]) if segment_ids else None
        extra_segments = [self._segment_row(segment_id) for segment_id in segment_ids[1:]]
        extra_segments = [segment for segment in extra_segments if segment is not None]
        return self._respond(question=question, focus_segment=focus_segment, extra_segments=extra_segments)

    def _respond(
        self,
        *,
        question: str,
        focus_segment: dict[str, Any] | None,
        extra_segments: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        extra_segments = extra_segments or []
        if focus_segment is None:
            return {
                "answer": "There is not enough grounded telemetry context to answer yet.",
                "evidence_segment_ids": [],
                "plan_code": {},
            }

        neighbors = self.nearest_neighbors(focus_segment["segment_id"])
        segment_summary = {
            "segment_id": focus_segment["segment_id"],
            "summary_payload": json_loads(focus_segment.get("summary_payload_json"), default={}),
            "plan_code": json_loads(focus_segment.get("plan_code_json"), default={}),
            "future_progress_3s": float(focus_segment.get("future_progress_3s", 0.0)),
            "future_return_3s": float(focus_segment.get("future_return_3s", 0.0)),
            "offtrack_next_3s": float(focus_segment.get("offtrack_next_3s", 0.0)),
        }
        evidence_segments = [focus_segment] + extra_segments + neighbors
        response = self.runtime.generate_coach_response(
            question=question,
            segment_summary=segment_summary,
            evidence_segments=evidence_segments,
        )
        interaction = CoachInteractionRecord(
            interaction_id=str(uuid.uuid4()),
            run_id=str(focus_segment.get("run_id", "")),
            episode=int(focus_segment.get("episode", 0)),
            segment_ids=[segment_summary["segment_id"]],
            question=question,
            answer=response["answer"],
            plan_code_used=response["plan_code"],
            evidence_segment_ids=response["evidence_segment_ids"],
        )
        with self.config.interaction_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(interaction.to_row(), separators=(",", ":")) + "\n")
        return response
