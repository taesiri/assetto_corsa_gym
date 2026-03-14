import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Build a curated seed-lap pack from the best live episodes.")
    parser.add_argument("--output-dir", required=True, help="Destination directory for curated laps.")
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="Run directory that contains episodes_stats.csv and laps/.",
    )
    parser.add_argument("--per-run-limit", type=int, default=8, help="Max episodes kept per run.")
    parser.add_argument("--total-limit", type=int, default=32, help="Max total episodes kept.")
    parser.add_argument("--min-reward", type=float, default=80.0)
    parser.add_argument("--min-speed", type=float, default=18.0)
    parser.add_argument("--max-gap", type=float, default=1.4)
    parser.add_argument("--max-overspeed", type=float, default=6.0)
    return parser.parse_args()


def get_float(row, key, default=0.0):
    value = row.get(key, default)
    if pd.isna(value):
        return default
    return float(value)


def episode_score(row):
    return (
        (0.06 * get_float(row, "ep_reward"))
        + (6.0 * get_float(row, "speed_mean"))
        + (0.12 * get_float(row, "ep_steps"))
        - (18.0 * get_float(row, "gap_abs_mean"))
        - (12.0 * get_float(row, "overspeed_mean"))
        - (80.0 * get_float(row, "out_of_track_frac"))
    )


def column_or_default(frame: pd.DataFrame, column: str, default: float):
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(default)
    return pd.Series([default] * len(frame), index=frame.index, dtype="float64")


def load_candidates(run_dir: Path, per_run_limit: int, min_reward: float, min_speed: float, max_gap: float, max_overspeed: float):
    episodes_path = run_dir / "episodes_stats.csv"
    laps_dir = run_dir / "laps"
    if not episodes_path.exists() or not laps_dir.exists():
        return []

    episodes_df = pd.read_csv(episodes_path)
    lap_files = sorted(laps_dir.glob("*_states.parquet"))
    aligned = min(len(episodes_df), len(lap_files))
    if aligned == 0:
        return []

    episodes_df = episodes_df.iloc[:aligned].copy()
    episodes_df["lap_file"] = [path.name for path in lap_files[:aligned]]
    episodes_df["run_dir"] = run_dir.as_posix()
    episodes_df["episode_score"] = episodes_df.apply(episode_score, axis=1)

    gap_series = column_or_default(episodes_df, "gap_abs_mean", 99.0)
    overspeed_series = column_or_default(episodes_df, "overspeed_mean", 0.0)

    filtered = episodes_df[
        (pd.to_numeric(episodes_df["ep_reward"], errors="coerce") >= min_reward)
        & (pd.to_numeric(episodes_df["speed_mean"], errors="coerce") >= min_speed)
        & (gap_series <= max_gap)
        & (overspeed_series <= max_overspeed)
    ].copy()

    filtered = filtered.sort_values("episode_score", ascending=False).head(per_run_limit)
    return filtered.to_dict(orient="records")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    laps_output = output_dir / "laps"
    laps_output.mkdir(parents=True, exist_ok=True)

    candidates = []
    static_info_path = None
    for run_dir_str in args.run_dir:
        run_dir = Path(run_dir_str).resolve()
        run_candidates = load_candidates(
            run_dir,
            args.per_run_limit,
            args.min_reward,
            args.min_speed,
            args.max_gap,
            args.max_overspeed,
        )
        candidates.extend(run_candidates)
        candidate_static = run_dir / "laps" / "static_info.json"
        if static_info_path is None and candidate_static.exists():
            static_info_path = candidate_static

    candidates = sorted(candidates, key=lambda row: row["episode_score"], reverse=True)[: args.total_limit]
    copied = []
    for rank, row in enumerate(candidates, start=1):
        source_run = Path(row["run_dir"])
        source_path = source_run / "laps" / row["lap_file"]
        if not source_path.exists():
            continue
        target_name = f"{rank:02d}_{source_run.name}_{row['lap_file']}"
        target_path = laps_output / target_name
        shutil.copy2(source_path, target_path)
        copied.append(
            {
                "rank": rank,
                "source_run_dir": source_run.as_posix(),
                "source_file": source_path.name,
                "target_file": target_name,
                "ep_reward": get_float(row, "ep_reward"),
                "speed_mean": get_float(row, "speed_mean"),
                "gap_abs_mean": get_float(row, "gap_abs_mean"),
                "overspeed_mean": get_float(row, "overspeed_mean"),
                "episode_score": get_float(row, "episode_score"),
            }
        )

    if static_info_path is not None:
        shutil.copy2(static_info_path, laps_output / "static_info.json")

    manifest = {
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "source_runs": args.run_dir,
        "per_run_limit": args.per_run_limit,
        "total_limit": args.total_limit,
        "count": len(copied),
        "laps": copied,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
