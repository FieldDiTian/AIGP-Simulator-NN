"""Convert FlightSim JSONL dynamics logs into ROS 2 rosbags."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from aigp_bag_tools.conversions import expand_input_paths
from aigp_bag_tools.rosbag_io import (
    build_metadata_message,
    build_row_messages,
    create_writer,
    register_default_topics,
    write_message,
)


def main():
    parser = argparse.ArgumentParser(description="Convert AI-GP FlightSim JSONL logs into ROS 2 rosbags.")
    parser.add_argument("--input", nargs="+", required=True, help="JSONL files, directories, or catalog JSON files.")
    parser.add_argument("--output-root", default="logs/rosbags", help="Directory for per-run bag folders.")
    parser.add_argument("--storage-id", default="sqlite3", choices=["sqlite3", "mcap"])
    parser.add_argument("--include-raw-json", action="store_true", help="Write /aigp/debug/raw_json.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing bag directories.")
    parser.add_argument(
        "--storage-time",
        choices=["row", "source"],
        default="row",
        help=(
            "rosbag storage timestamp policy. 'row' keeps all snapshot topics aligned by JSONL row; "
            "'source' stores each topic at its own sensor/source timestamp."
        ),
    )
    args = parser.parse_args()

    paths = expand_input_paths(args.input)
    if not paths:
        raise RuntimeError("No JSONL inputs found.")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for path in paths:
        summaries.append(
            convert_jsonl_to_bag(
                path,
                output_root=output_root,
                storage_id=args.storage_id,
                include_raw_json=args.include_raw_json,
                overwrite=args.overwrite,
                storage_time_mode=args.storage_time,
            )
        )
    print(json.dumps({"converted": summaries}, indent=2), flush=True)


def convert_jsonl_to_bag(
    path: Path,
    *,
    output_root: Path,
    storage_id="sqlite3",
    include_raw_json=False,
    overwrite=False,
    storage_time_mode="row",
) -> dict:
    metadata_row, dynamics_lines = read_jsonl_lines(path)
    if not dynamics_lines:
        raise RuntimeError(f"No dynamics rows found in {path}")

    first_row = dynamics_lines[0][1]
    run_id = str((metadata_row or {}).get("run_id") or first_row.get("run_id") or path.stem)
    origin_wall_ns = int(first_row["t_wall_ns"])
    output_dir = output_root / run_id
    if output_dir.exists():
        if not overwrite:
            raise RuntimeError(f"Bag already exists: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)

    ros, writer = create_writer(output_dir, storage_id=storage_id)
    register_default_topics(writer, ros, include_raw_json=include_raw_json)
    topic_counts = Counter()

    metadata_msg = build_metadata_message(ros, metadata_row, run_id, origin_wall_ns)
    write_message(writer, ros, "/aigp/run/metadata", metadata_msg, 0)
    topic_counts["/aigp/run/metadata"] += 1

    for line_index, row, raw_line in dynamics_lines:
        for topic, msg, stamp_ns in build_row_messages(
            ros,
            row,
            origin_wall_ns,
            line_index=line_index,
            raw_line=raw_line,
            storage_time_mode=storage_time_mode,
            include_raw_json=include_raw_json,
        ):
            write_message(writer, ros, topic, msg, stamp_ns)
            topic_counts[topic] += 1

    return {
        "input": str(path),
        "output": str(output_dir),
        "run_id": run_id,
        "dynamics_rows": len(dynamics_lines),
        "origin_t_wall_ns": origin_wall_ns,
        "storage_id": storage_id,
        "storage_time": storage_time_mode,
        "topics": dict(sorted(topic_counts.items())),
    }


def read_jsonl_lines(path: Path):
    metadata_row = None
    dynamics_lines = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("sample_type") == "metadata":
                metadata_row = row
            elif row.get("sample_type") == "dynamics":
                if row.get("t_wall_ns") is None:
                    continue
                dynamics_lines.append((line_index, row, line))
    return metadata_row, dynamics_lines


if __name__ == "__main__":
    main()
