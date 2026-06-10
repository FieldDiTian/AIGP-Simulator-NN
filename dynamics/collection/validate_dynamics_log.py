import argparse
import json
import math
import sys
from pathlib import Path


def vector_norm(values):
    return math.sqrt(sum(float(value) * float(value) for value in values))


def main():
    parser = argparse.ArgumentParser(description="Validate that a raw dynamics jsonl contains moving flight data.")
    parser.add_argument("log_path")
    parser.add_argument("--min-complete-rows", type=int, default=120)
    parser.add_argument("--min-displacement-m", type=float, default=0.5)
    parser.add_argument("--min-speed-mps", type=float, default=0.3)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    path = Path(args.log_path)
    if not path.exists():
        raise FileNotFoundError(path)

    complete_rows = []
    for line in path.open("r", encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("sample_type") != "dynamics":
            continue
        if not all(row.get(key) is not None for key in ("action", "odometry", "imu", "attitude")):
            continue
        complete_rows.append(row)

    displacement_m = 0.0
    max_speed_mps = 0.0
    if complete_rows:
        p0 = complete_rows[0]["odometry"]["p_ned"]
        p1 = complete_rows[-1]["odometry"]["p_ned"]
        displacement_m = vector_norm([float(p1[i]) - float(p0[i]) for i in range(3)])
        for row in complete_rows:
            local_position = row.get("local_position_ned") or {}
            velocity = local_position.get("v_ned") or row["odometry"].get("v") or [0.0, 0.0, 0.0]
            max_speed_mps = max(max_speed_mps, vector_norm(velocity))

    valid = (
        len(complete_rows) >= args.min_complete_rows
        and (
            displacement_m >= args.min_displacement_m
            or max_speed_mps >= args.min_speed_mps
        )
    )
    result = {
        "log_path": str(path),
        "valid": valid,
        "complete_rows": len(complete_rows),
        "displacement_m": displacement_m,
        "max_speed_mps": max_speed_mps,
        "min_complete_rows": args.min_complete_rows,
        "min_displacement_m": args.min_displacement_m,
        "min_speed_mps": args.min_speed_mps,
    }
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(
            f"valid={valid} complete_rows={len(complete_rows)} "
            f"displacement_m={displacement_m:.3f} max_speed_mps={max_speed_mps:.3f}"
        )

    return 0 if valid else 2


if __name__ == "__main__":
    sys.exit(main())
