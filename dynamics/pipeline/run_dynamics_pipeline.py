import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXE = ROOT / "AIGP_3364" / "FlightSim.exe"
DEFAULT_PROFILES = [
    "forward_flight",
    "left_turn",
    "right_turn",
    "climb_descend",
]


def main():
    parser = argparse.ArgumentParser(
        description="Run FlightSim collection -> wrapper dataset -> CUDA MLP -> eval/coverage pipeline."
    )
    parser.add_argument("--profiles", nargs="+", default=DEFAULT_PROFILES)
    parser.add_argument("--duration-s", type=float, default=8.0)
    parser.add_argument("--run-prefix", default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--checkpoint-name", default=None)
    parser.add_argument("--coverage-name", default=None)
    parser.add_argument("--log-dir", default=str(ROOT / "logs" / "raw"))
    parser.add_argument("--processed-root", default=str(ROOT / "logs" / "processed"))
    parser.add_argument("--checkpoint-root", default=str(ROOT / "checkpoints"))
    parser.add_argument("--coverage-root", default=str(ROOT / "logs" / "coverage"))
    parser.add_argument("--rejected-dir", default=str(ROOT / "logs" / "rejected"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=14550)
    parser.add_argument("--skip-launch", action="store_true")
    parser.add_argument("--manual-ready-per-profile", action="store_true")
    parser.add_argument("--launch-mode", choices=["manual-ready", "ui-auto"], default="ui-auto")
    parser.add_argument("--exe", default=str(DEFAULT_EXE))
    parser.add_argument("--launch-timeout-s", type=float, default=150.0)
    parser.add_argument("--launch-attempts", type=int, default=1)
    parser.add_argument("--profile-attempts", type=int, default=1)
    parser.add_argument("--restart-delay-s", type=float, default=2.0)
    parser.add_argument("--save-ui-screenshots", action="store_true")
    parser.add_argument("--relaunch-per-profile", action="store_true")
    parser.add_argument("--close-after-profile", action="store_true")
    parser.add_argument("--close-after-launch-ready", action="store_true")
    parser.add_argument("--heartbeat-timeout-s", type=float, default=45.0)
    parser.add_argument("--telemetry-timeout-s", type=float, default=45.0)
    parser.add_argument("--skip-flight-response-probe", action="store_true")
    parser.add_argument("--probe-duration-s", type=float, default=1.5)
    parser.add_argument("--probe-min-displacement-m", type=float, default=0.05)
    parser.add_argument("--probe-min-speed-mps", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--hz", type=float, default=60.0)
    parser.add_argument("--max-rate", type=float, default=1.0)
    parser.add_argument("--no-actuator", action="store_true")
    parser.add_argument("--no-imu", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--coverage-min-samples-per-bin", type=int, default=20)
    parser.add_argument("--rollout-horizons-s", nargs="+", type=float, default=[1.0])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_prefix = args.run_prefix or f"pipeline_{stamp}"
    dataset_name = args.dataset_name or run_prefix
    checkpoint_name = args.checkpoint_name or run_prefix
    coverage_name = args.coverage_name or run_prefix

    log_dir = Path(args.log_dir)
    processed_dir = Path(args.processed_root) / dataset_name
    checkpoint_dir = Path(args.checkpoint_root) / checkpoint_name
    coverage_dir = Path(args.coverage_root) / coverage_name
    rejected_dir = Path(args.rejected_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    commands = []
    raw_logs = []
    validation_results = []

    if args.skip_launch:
        print("skip-launch: expecting FlightSim to already be in the flyable HUD.", flush=True)
    elif args.manual_ready_per_profile:
        print("manual-ready-per-profile: each profile waits for an already flyable FlightSim HUD.", flush=True)
    elif not args.relaunch_per_profile:
        run_launch(args, commands)

    for index, profile in enumerate(args.profiles, start=1):
        profile_done = False
        attempts = max(1, int(args.profile_attempts))
        for profile_attempt in range(1, attempts + 1):
            if args.manual_ready_per_profile:
                wait_manual_ready(args, commands, profile, index, profile_attempt, attempts)
            elif (args.relaunch_per_profile or profile_attempt > 1) and not args.skip_launch:
                run_launch(args, commands)

            run_id = profile_run_id(run_prefix, index, profile, profile_attempt, attempts)
            raw_log = log_dir / f"{run_id}.jsonl"
            collect_cmd = [
                sys.executable,
                str(ROOT / "dynamics" / "collection" / "collect_identification_data.py"),
                "--profile",
                profile,
                "--duration-s",
                str(args.duration_s),
                "--host",
                args.host,
                "--port",
                str(args.port),
                "--heartbeat-timeout-s",
                str(args.heartbeat_timeout_s),
                "--telemetry-timeout-s",
                str(args.telemetry_timeout_s),
                "--probe-duration-s",
                str(args.probe_duration_s),
                "--probe-min-displacement-m",
                str(args.probe_min_displacement_m),
                "--probe-min-speed-mps",
                str(args.probe_min_speed_mps),
                "--log-dir",
                str(log_dir),
                "--run-id",
                run_id,
                "--seed",
                str(args.seed + index - 1 + (profile_attempt - 1) * 1000),
            ]
            if args.skip_flight_response_probe:
                collect_cmd.append("--skip-flight-response-probe")

            try:
                run_command(collect_cmd, commands, args.dry_run)
            except subprocess.CalledProcessError as exc:
                validation_results.append({
                    "profile": profile,
                    "run_id": run_id,
                    "log_path": str(raw_log),
                    "validation": {
                        "valid": False,
                        "error": f"collection command failed with exit code {exc.returncode}",
                    },
                })
                close_flightsim_after_failed_attempt(args, commands)
                if profile_attempt < attempts:
                    continue
                break

            validate_cmd = [
                sys.executable,
                str(ROOT / "dynamics" / "collection" / "validate_dynamics_log.py"),
                str(raw_log),
                "--json",
            ]
            validation = run_json_command(validate_cmd, commands, args.dry_run)
            validation_results.append({
                "profile": profile,
                "run_id": run_id,
                "log_path": str(raw_log),
                "validation": validation,
            })
            if args.dry_run:
                raw_logs.append(raw_log)
                profile_done = True
                break

            if not validation.get("valid"):
                rejected_path = rejected_dir / f"{raw_log.stem}.invalid.jsonl"
                if raw_log.exists():
                    shutil.move(str(raw_log), str(rejected_path))
                print(f"Rejected invalid log: {raw_log} -> {rejected_path}", flush=True)
                close_flightsim_after_failed_attempt(args, commands)
                if profile_attempt < attempts:
                    continue
                break

            raw_logs.append(raw_log)
            profile_done = True
            if args.close_after_profile or args.relaunch_per_profile:
                close_flightsim(commands, args.dry_run)
            break

        if not profile_done:
            print(f"Profile failed after {attempts} attempt(s): {profile}", flush=True)

    if not raw_logs:
        raise RuntimeError("No valid raw logs were collected; dataset/training were not run.")

    build_cmd = [
        sys.executable,
        str(ROOT / "dynamics" / "dataset" / "build_dataset.py"),
        "--input",
        *[str(path) for path in raw_logs],
        "--output-dir",
        str(processed_dir),
        "--k",
        str(args.k),
        "--hz",
        str(args.hz),
        "--max-rate",
        str(args.max_rate),
    ]
    if args.no_actuator:
        build_cmd.append("--no-actuator")
    if args.no_imu:
        build_cmd.append("--no-imu")
    run_command(build_cmd, commands, args.dry_run)

    checkpoint_path = checkpoint_dir / "best_val_model.pt"
    if not args.skip_train:
        train_cmd = [
            sys.executable,
            str(ROOT / "dynamics" / "training" / "train_mlp_dynamics.py"),
            "--data-dir",
            str(processed_dir),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--config-dir",
            str(ROOT / "configs"),
            "--hidden-dim",
            str(args.hidden_dim),
            "--num-layers",
            str(args.num_layers),
            "--batch-size",
            str(args.batch_size),
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--device",
            args.device,
        ]
        run_command(train_cmd, commands, args.dry_run)

    if not args.skip_train:
        one_step_cmd = [
            sys.executable,
            str(ROOT / "dynamics" / "evaluation" / "eval_one_step.py"),
            "--checkpoint",
            str(checkpoint_path),
            "--raw-input",
            *[str(path) for path in raw_logs],
            "--device",
            args.device,
        ]
        if args.no_actuator:
            one_step_cmd.append("--no-actuator")
        if args.no_imu:
            one_step_cmd.append("--no-imu")
        run_command(one_step_cmd, commands, args.dry_run)

        rollout_cmd = [
            sys.executable,
            str(ROOT / "dynamics" / "evaluation" / "eval_rollout.py"),
            "--checkpoint",
            str(checkpoint_path),
            "--raw-input",
            *[str(path) for path in raw_logs],
            "--horizons-s",
            *[str(value) for value in args.rollout_horizons_s],
            "--device",
            args.device,
        ]
        if args.no_actuator:
            rollout_cmd.append("--no-actuator")
        if args.no_imu:
            rollout_cmd.append("--no-imu")
        run_command(rollout_cmd, commands, args.dry_run)

    coverage_cmd = [
        sys.executable,
        str(ROOT / "dynamics" / "evaluation" / "analyze_coverage.py"),
        "--input",
        *[str(path) for path in raw_logs],
        "--output-dir",
        str(coverage_dir),
        "--k",
        str(args.k),
        "--hz",
        str(args.hz),
        "--max-rate",
        str(args.max_rate),
        "--min-samples-per-bin",
        str(args.coverage_min_samples_per_bin),
    ]
    if args.no_actuator:
        coverage_cmd.append("--no-actuator")
    if args.no_imu:
        coverage_cmd.append("--no-imu")
    if not args.skip_train:
        coverage_cmd.extend(["--checkpoint", str(checkpoint_path), "--device", args.device])
    run_command(coverage_cmd, commands, args.dry_run)

    manifest = {
        "run_prefix": run_prefix,
        "profiles": args.profiles,
        "raw_logs": [str(path) for path in raw_logs],
        "processed_dir": str(processed_dir),
        "checkpoint_dir": str(checkpoint_dir) if not args.skip_train else None,
        "coverage_dir": str(coverage_dir),
        "validation_results": validation_results,
        "commands": commands,
        "uses_wrapper_from_stage": "dataset/evaluation/coverage",
        "raw_collection_uses_wrapper": False,
    }
    if not args.dry_run:
        manifest_path = processed_dir / "pipeline_manifest.json"
        processed_dir.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        print(f"Pipeline manifest: {manifest_path}", flush=True)
    print(json.dumps(manifest, indent=2), flush=True)


def run_launch(args, commands):
    launch_cmd = [
        sys.executable,
        str(ROOT / "dynamics" / "collection" / "launch_flightsim.py"),
        "--mode",
        args.launch_mode,
        "--exe",
        args.exe,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--timeout-s",
        str(args.launch_timeout_s),
        "--attempts",
        str(args.launch_attempts),
        "--restart-delay-s",
        str(args.restart_delay_s),
        "--ready-signal",
        "telemetry",
    ]
    if args.save_ui_screenshots:
        launch_cmd.append("--save-screenshots")
    if args.close_after_launch_ready:
        launch_cmd.append("--close-after-ready")
    run_command(launch_cmd, commands, args.dry_run)


def wait_manual_ready(args, commands, profile, index, profile_attempt=1, profile_attempts=1):
    print(
        f"\nManual step required for profile {index}: {profile} "
        f"(attempt {profile_attempt}/{profile_attempts})\n"
        "Put FlightSim into the flyable HUD, then this pipeline will continue after telemetry is detected.",
        flush=True,
    )
    launch_cmd = [
        sys.executable,
        str(ROOT / "dynamics" / "collection" / "launch_flightsim.py"),
        "--mode",
        "manual-ready",
        "--no-start",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--timeout-s",
        str(args.launch_timeout_s),
        "--attempts",
        str(args.launch_attempts),
        "--restart-delay-s",
        str(args.restart_delay_s),
        "--ready-signal",
        "telemetry",
    ]
    run_command(launch_cmd, commands, args.dry_run)


def profile_run_id(run_prefix, index, profile, profile_attempt, profile_attempts):
    base = f"{run_prefix}_{index:02d}_{profile}"
    if profile_attempts <= 1:
        return base
    return f"{base}_try{profile_attempt:02d}"


def close_flightsim_after_failed_attempt(args, commands):
    if args.skip_launch:
        return
    close_flightsim(commands, args.dry_run)
    if args.restart_delay_s > 0 and not args.dry_run:
        import time
        time.sleep(args.restart_delay_s)


def close_flightsim(commands, dry_run):
    if sys.platform != "win32":
        return
    cmd = [
        "powershell",
        "-NoProfile",
        "-Command",
        (
            "Get-Process | Where-Object { $_.ProcessName -in "
            "@('FlightSim','DCGame-Win64-Shipping') } | Stop-Process -Force"
        ),
    ]
    run_command(cmd, commands, dry_run)


def run_json_command(cmd, commands, dry_run):
    printable = " ".join(quote_part(part) for part in cmd)
    commands.append(printable)
    print(f"\n$ {printable}", flush=True)
    if dry_run:
        return {"valid": True, "dry_run": True}
    result = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    output = result.stdout
    if result.returncode != 0 and not output.strip():
        raise subprocess.CalledProcessError(
            result.returncode,
            cmd,
            output=result.stdout,
            stderr=result.stderr,
        )
    try:
        return json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Command did not return JSON: {' '.join(cmd)}\n{output}") from exc


def run_command(cmd, commands, dry_run, capture=False):
    printable = " ".join(quote_part(part) for part in cmd)
    commands.append(printable)
    print(f"\n$ {printable}", flush=True)
    if dry_run:
        return "{}" if capture else None
    if capture:
        result = subprocess.run(
            cmd,
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.stdout:
            print(result.stdout, end="", flush=True)
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr, flush=True)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode,
                cmd,
                output=result.stdout,
                stderr=result.stderr,
            )
        return result.stdout

    subprocess.run(cmd, cwd=str(ROOT), check=True)
    return None


def quote_part(part):
    text = str(part)
    if any(ch.isspace() for ch in text):
        return f'"{text}"'
    return text


if __name__ == "__main__":
    main()
