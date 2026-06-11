"""Verify a converted AI-GP rosbag."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from aigp_bag_tools.jsonl_to_rosbag import read_jsonl_lines
from aigp_bag_tools.rosbag_io import TOPIC_TYPES, read_bag_messages


REQUIRED_TOPICS = [
    "/aigp/raw/odom_ned",
    "/odom",
    "/tf",
    "/aigp/state/pose",
    "/aigp/state/twist",
    "/imu/data",
    "/aigp/control/attitude_target",
    "/aigp/control/body_rates_cmd",
    "/aigp/control/thrust_cmd",
    "/aigp/debug/sample_info",
]


def main():
    parser = argparse.ArgumentParser(description="Verify AI-GP ROS 2 bag topic structure and counts.")
    parser.add_argument("--bag", required=True, help="Bag directory.")
    parser.add_argument("--jsonl", default=None, help="Optional source JSONL for expected row counts.")
    parser.add_argument("--storage-id", default="sqlite3", choices=["sqlite3", "mcap"])
    parser.add_argument("--json", action="store_true", help="Print JSON only.")
    args = parser.parse_args()

    result = verify_bag(Path(args.bag), jsonl=Path(args.jsonl) if args.jsonl else None, storage_id=args.storage_id)
    if args.json:
        print(json.dumps(result, indent=2), flush=True)
    else:
        print(json.dumps(result, indent=2), flush=True)
    if not result["valid"]:
        raise SystemExit(1)


def verify_bag(bag_dir: Path, jsonl: Path | None = None, storage_id="sqlite3") -> dict:
    topic_counts = Counter()
    topic_types = {}
    last_stamp_by_topic = defaultdict(lambda: -1)
    monotonic_errors = []

    for topic, _msg, timestamp, type_name in read_bag_messages(bag_dir, storage_id=storage_id):
        topic_counts[topic] += 1
        topic_types[topic] = type_name
        if timestamp < last_stamp_by_topic[topic]:
            monotonic_errors.append(
                {
                    "topic": topic,
                    "previous": last_stamp_by_topic[topic],
                    "current": timestamp,
                }
            )
        last_stamp_by_topic[topic] = timestamp

    missing_topics = [topic for topic in REQUIRED_TOPICS if topic not in topic_counts]
    type_errors = [
        {
            "topic": topic,
            "expected": expected,
            "actual": topic_types.get(topic),
        }
        for topic, expected in TOPIC_TYPES.items()
        if topic in topic_types and topic_types.get(topic) != expected
    ]

    expected = {}
    count_errors = []
    if jsonl is not None:
        metadata_row, dynamics_lines = read_jsonl_lines(jsonl)
        rows = [row for _line_index, row, _raw_line in dynamics_lines]
        dynamics_rows = len(rows)
        expected["dynamics_rows"] = dynamics_rows
        expected_counts = expected_counts_from_rows(rows)
        expected["topic_counts"] = expected_counts
        for topic, expected_count in expected_counts.items():
            actual = topic_counts.get(topic, 0)
            if actual != expected_count:
                count_errors.append({"topic": topic, "expected": expected_count, "actual": actual})
        metadata_expected = 1 if metadata_row is not None else 1
        if topic_counts.get("/aigp/run/metadata", 0) != metadata_expected:
            count_errors.append(
                {
                    "topic": "/aigp/run/metadata",
                    "expected": metadata_expected,
                    "actual": topic_counts.get("/aigp/run/metadata", 0),
                }
            )

    valid = not missing_topics and not type_errors and not monotonic_errors and not count_errors
    return {
        "valid": valid,
        "bag": str(bag_dir),
        "topics": dict(sorted(topic_counts.items())),
        "topic_types": dict(sorted(topic_types.items())),
        "missing_topics": missing_topics,
        "type_errors": type_errors,
        "monotonic_errors": monotonic_errors[:20],
        "count_errors": count_errors,
        "expected": expected,
    }


def expected_counts_from_rows(rows: list[dict]) -> dict:
    counts = Counter()
    for row in rows:
        if row.get("odometry"):
            for topic in (
                "/aigp/raw/odom_ned",
                "/odom",
                "/tf",
                "/aigp/state/pose",
                "/aigp/state/twist",
                "/aigp/state/attitude_rpy",
                "/aigp/state/gravity_body",
            ):
                counts[topic] += 1
        if row.get("local_position_ned"):
            counts["/aigp/raw/local_position_ned"] += 1
        if row.get("imu"):
            for topic in ("/imu/data", "/aigp/raw/imu_frd", "/aigp/state/linear_acceleration"):
                counts[topic] += 1
        if row.get("attitude"):
            for topic in ("/aigp/raw/attitude_rpy_frd", "/aigp/state/angular_velocity"):
                counts[topic] += 1
        if row.get("action"):
            for topic in (
                "/aigp/control/attitude_target",
                "/aigp/control/body_rates_cmd",
                "/aigp/control/thrust_cmd",
            ):
                counts[topic] += 1
        if row.get("actuator_output"):
            counts["/aigp/actuator_output"] += 1
        counts["/aigp/run/reset_counter"] += 1
        counts["/aigp/collision"] += 1
        counts["/aigp/debug/sample_info"] += 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    main()
