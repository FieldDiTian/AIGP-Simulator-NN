import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynamics.collection.collect_identification_data import PROFILE_METADATA


def main():
    parser = argparse.ArgumentParser(description="Create categorized manifests for raw dynamics jsonl logs.")
    parser.add_argument("--input", nargs="+", default=[str(ROOT / "logs" / "raw")])
    parser.add_argument("--output-dir", default=str(ROOT / "logs" / "catalog"))
    parser.add_argument("--catalog-name", default=None)
    args = parser.parse_args()

    paths = expand_inputs(args.input)
    catalog_name = args.catalog_name or datetime.now().strftime("catalog_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / catalog_name
    categories_dir = output_dir / "categories"
    profiles_dir = output_dir / "profiles"
    categories_dir.mkdir(parents=True, exist_ok=True)
    profiles_dir.mkdir(parents=True, exist_ok=True)

    entries = [inspect_log(path) for path in paths]
    entries = [entry for entry in entries if entry is not None]
    entries.sort(key=lambda entry: (entry["category"], entry["profile"], entry["path"]))

    by_category = defaultdict(list)
    by_profile = defaultdict(list)
    for entry in entries:
        by_category[entry["category"]].append(entry)
        by_profile[entry["profile"]].append(entry)

    write_json(output_dir / "all_runs.json", entries)
    write_json(output_dir / "summary.json", {
        "catalog_name": catalog_name,
        "created_at": datetime.now().isoformat(),
        "runs": len(entries),
        "categories": {key: len(value) for key, value in sorted(by_category.items())},
        "profiles": {key: len(value) for key, value in sorted(by_profile.items())},
    })
    for category, category_entries in sorted(by_category.items()):
        write_json(categories_dir / f"{safe_name(category)}.json", category_entries)
    for profile, profile_entries in sorted(by_profile.items()):
        write_json(profiles_dir / f"{safe_name(profile)}.json", profile_entries)

    readme = [
        "# Raw Dynamics Log Catalog",
        "",
        "This catalog groups raw jsonl paths by semantic collection category and profile.",
        "It stores paths and metadata only; raw logs remain in logs/raw.",
        "",
        "Files:",
        "",
        "- all_runs.json: every indexed run",
        "- summary.json: counts by category/profile",
        "- categories/*.json: paths grouped by collection category",
        "- profiles/*.json: paths grouped by exact profile",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(json.dumps({
        "catalog_dir": str(output_dir),
        "runs": len(entries),
        "categories": {key: len(value) for key, value in sorted(by_category.items())},
    }, indent=2), flush=True)


def expand_inputs(items):
    paths = []
    for item in items:
        path = Path(item)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.jsonl")))
        else:
            paths.append(path)
    return paths


def inspect_log(path):
    metadata = None
    dynamics_rows = 0
    first_dynamics_t_wall_ns = None
    last_dynamics_t_wall_ns = None
    run_id = path.stem
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("sample_type") == "metadata":
                metadata = row.get("metadata") or {}
                run_id = row.get("run_id", run_id)
                continue
            if row.get("sample_type") == "dynamics":
                dynamics_rows += 1
                run_id = row.get("run_id", run_id)
                t_wall_ns = row.get("t_wall_ns")
                if t_wall_ns is not None:
                    first_dynamics_t_wall_ns = (
                        int(t_wall_ns)
                        if first_dynamics_t_wall_ns is None
                        else min(first_dynamics_t_wall_ns, int(t_wall_ns))
                    )
                    last_dynamics_t_wall_ns = (
                        int(t_wall_ns)
                        if last_dynamics_t_wall_ns is None
                        else max(last_dynamics_t_wall_ns, int(t_wall_ns))
                    )

    if metadata is None and dynamics_rows == 0:
        return None

    profile = (metadata or {}).get("profile") or profile_from_run_id(run_id)
    profile_info = PROFILE_METADATA.get(profile, {})
    category = (metadata or {}).get("profile_category") or profile_info.get("category") or "unknown"
    coverage_goal = (metadata or {}).get("coverage_goal") or profile_info.get("coverage_goal") or "unknown"
    duration_s = None
    if first_dynamics_t_wall_ns is not None and last_dynamics_t_wall_ns is not None:
        duration_s = (last_dynamics_t_wall_ns - first_dynamics_t_wall_ns) / 1e9

    return {
        "path": str(path.resolve()),
        "file": path.name,
        "run_id": run_id,
        "profile": profile,
        "category": category,
        "coverage_goal": coverage_goal,
        "metadata": metadata or {},
        "dynamics_rows": dynamics_rows,
        "duration_s": duration_s,
    }


def profile_from_run_id(run_id):
    for profile in sorted(PROFILE_METADATA, key=len, reverse=True):
        if profile in run_id:
            return profile
    return "unknown"


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)


def safe_name(value):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


if __name__ == "__main__":
    main()
