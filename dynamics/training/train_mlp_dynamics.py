import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    return data["x"].astype(np.float32), data["y"].astype(np.float32)


def load_npz_full(path):
    data = np.load(path, allow_pickle=True)
    return {
        "x": data["x"].astype(np.float32),
        "y": data["y"].astype(np.float32),
        "run_id": data["run_id"],
        "segment_idx": data["segment_idx"],
        "time_s": data["time_s"].astype(np.float64),
    }


def normalize(x, mean, std):
    return (x - mean) / std


def main():
    parser = argparse.ArgumentParser(description="Train a history-stacked MLP drone dynamics model.")
    parser.add_argument("--data-dir", default=str(ROOT / "logs" / "processed"))
    parser.add_argument("--checkpoint-dir", default=str(ROOT / "checkpoints"))
    parser.add_argument("--config-dir", default=str(ROOT / "configs"))
    parser.add_argument("--model-type", choices=["mlp", "gru"], default="mlp")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--rollout-loss-weight", type=float, default=0.0)
    parser.add_argument("--rollout-steps", type=int, default=10)
    parser.add_argument("--rollout-batch-size", type=int, default=None)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from dynamics.models.mlp_dynamics import build_model
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = Path(args.data_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    config_dir = Path(args.config_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    train_full = load_npz_full(data_dir / "train.npz")
    val_full = load_npz_full(data_dir / "val.npz")
    x_train, y_train = train_full["x"], train_full["y"]
    x_val, y_val = val_full["x"], val_full["y"]
    if len(x_train) == 0:
        raise RuntimeError("No training samples found. Run build_dataset.py first.")
    if len(x_val) == 0:
        x_val, y_val = x_train, y_train

    with open(data_dir / "normalization_stats.json", "r", encoding="utf-8") as handle:
        stats = json.load(handle)
    x_mean = np.asarray(stats["x_mean"], dtype=np.float32)
    x_std = np.asarray(stats["x_std"], dtype=np.float32)
    y_mean = np.asarray(stats["y_mean"], dtype=np.float32)
    y_std = np.asarray(stats["y_std"], dtype=np.float32)

    x_train_n = normalize(x_train, x_mean, x_std)
    y_train_n = normalize(y_train, y_mean, y_std)
    x_val_n = normalize(x_val, x_mean, x_std)
    y_val_n = normalize(y_val, y_mean, y_std)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train_n), torch.from_numpy(y_train_n)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val_n), torch.from_numpy(y_val_n)),
        batch_size=args.batch_size,
        shuffle=False,
    )

    config = {
        "model_type": args.model_type,
        "input_dim": int(stats["input_dim"]),
        "output_dim": int(stats["output_dim"]),
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "k": int(stats["k"]),
        "state_dim": int(stats["state_dim"]),
        "action_dim": int(stats["action_dim"]),
        "use_actuator": bool(stats["use_actuator"]),
        "use_imu": bool(stats.get("use_imu", True)),
    }
    model = build_model(config).to(args.device)
    if args.init_checkpoint:
        init_checkpoint = torch.load(args.init_checkpoint, map_location=args.device)
        init_config = init_checkpoint["model_config"]
        if str(init_config.get("model_type", "mlp")) != args.model_type:
            raise ValueError(
                f"init checkpoint model_type={init_config.get('model_type', 'mlp')} "
                f"does not match requested model_type={args.model_type}"
            )
        if int(init_config["input_dim"]) != config["input_dim"]:
            raise ValueError(
                f"init checkpoint input_dim={init_config['input_dim']} "
                f"does not match dataset input_dim={config['input_dim']}"
            )
        model.load_state_dict(init_checkpoint["model_state_dict"])
        print(f"Loaded init checkpoint: {args.init_checkpoint}", flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    stale_epochs = 0
    curves = []
    best_path = checkpoint_dir / "best_val_model.pt"
    rollout_batch_size = args.rollout_batch_size or max(64, args.batch_size // 4)
    rollout_train_loader = None
    rollout_val_loader = None
    tensors_for_rollout = None
    if args.rollout_loss_weight > 0.0:
        train_seq = build_sequence_indices(train_full, args.rollout_steps, float(stats["dt"]))
        val_seq = build_sequence_indices(val_full, args.rollout_steps, float(stats["dt"]))
        if len(train_seq) == 0:
            raise RuntimeError("No contiguous training sequences found for rollout loss.")
        rollout_train_loader = DataLoader(
            TensorDataset(torch.from_numpy(train_seq)),
            batch_size=rollout_batch_size,
            shuffle=True,
        )
        if len(val_seq) > 0:
            rollout_val_loader = DataLoader(
                TensorDataset(torch.from_numpy(val_seq)),
                batch_size=rollout_batch_size,
                shuffle=False,
            )
        tensors_for_rollout = {
            "train_x": torch.from_numpy(x_train).to(args.device),
            "train_y": torch.from_numpy(y_train).to(args.device),
            "val_x": torch.from_numpy(x_val).to(args.device),
            "val_y": torch.from_numpy(y_val).to(args.device),
            "x_mean": torch.from_numpy(x_mean).to(args.device),
            "x_std": torch.from_numpy(x_std).to(args.device),
            "y_mean": torch.from_numpy(y_mean).to(args.device),
            "y_std": torch.from_numpy(y_std).to(args.device),
        }
        print(
            f"rollout_loss enabled: steps={args.rollout_steps} "
            f"train_sequences={len(train_seq)} val_sequences={len(val_seq)} "
            f"weight={args.rollout_loss_weight}",
            flush=True,
        )

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = run_epoch(model, train_loader, loss_fn, args.device, optimizer)
        rollout_train_loss = None
        if rollout_train_loader is not None:
            rollout_train_loss = run_rollout_epoch(
                model=model,
                loader=rollout_train_loader,
                arrays_x=tensors_for_rollout["train_x"],
                arrays_y=tensors_for_rollout["train_y"],
                stats_t=tensors_for_rollout,
                model_config=config,
                rollout_weight=args.rollout_loss_weight,
                optimizer=optimizer,
            )
        model.eval()
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, loss_fn, args.device, optimizer=None)
            rollout_val_loss = None
            if rollout_val_loader is not None:
                rollout_val_loss = run_rollout_epoch(
                    model=model,
                    loader=rollout_val_loader,
                    arrays_x=tensors_for_rollout["val_x"],
                    arrays_y=tensors_for_rollout["val_y"],
                    stats_t=tensors_for_rollout,
                    model_config=config,
                    rollout_weight=args.rollout_loss_weight,
                    optimizer=None,
                )

        selection_loss = val_loss
        if rollout_val_loss is not None:
            selection_loss = val_loss + args.rollout_loss_weight * rollout_val_loss

        curves.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "rollout_train_loss": rollout_train_loss,
            "rollout_val_loss": rollout_val_loss,
            "selection_loss": selection_loss,
        })
        msg = f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
        if rollout_train_loss is not None:
            msg += f" rollout_train_loss={rollout_train_loss:.6f}"
        if rollout_val_loss is not None:
            msg += f" rollout_val_loss={rollout_val_loss:.6f}"
        msg += f" selection_loss={selection_loss:.6f}"
        print(msg, flush=True)

        if selection_loss < best_val:
            best_val = selection_loss
            stale_epochs = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": config,
                "normalization_stats": stats,
            }, best_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print("Early stopping.", flush=True)
                break

    with open(checkpoint_dir / "training_curve.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "rollout_train_loss",
                "rollout_val_loss",
                "selection_loss",
            ],
        )
        writer.writeheader()
        writer.writerows(curves)

    with open(checkpoint_dir / "normalization_stats.json", "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    with open(config_dir / "train_mlp.yaml", "w", encoding="utf-8") as handle:
        for key, value in {
            **config,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "weight_decay": args.weight_decay,
            "early_stopping_patience": args.patience,
            "init_checkpoint": args.init_checkpoint,
            "rollout_loss_weight": args.rollout_loss_weight,
            "rollout_steps": args.rollout_steps,
            "rollout_batch_size": rollout_batch_size,
        }.items():
            handle.write(f"{key}: {value}\n")

    print(f"Saved best checkpoint to {best_path}", flush=True)


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    total_loss = 0.0
    total_count = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * len(x)
        total_count += len(x)
    return total_loss / max(total_count, 1)


def build_sequence_indices(split, rollout_steps, dt):
    if rollout_steps <= 1 or len(split["x"]) == 0:
        return np.zeros((0, max(1, rollout_steps)), dtype=np.int64)
    groups = {}
    for idx, (run_id, segment_idx, time_s) in enumerate(
        zip(split["run_id"], split["segment_idx"], split["time_s"])
    ):
        key = (str(run_id), int(segment_idx))
        groups.setdefault(key, []).append((float(time_s), idx))

    sequences = []
    max_gap = dt * 1.5
    for items in groups.values():
        items.sort()
        times = [item[0] for item in items]
        indices = [item[1] for item in items]
        for start in range(0, len(indices) - rollout_steps + 1):
            if times[start + rollout_steps - 1] - times[start] > max_gap * (rollout_steps - 1):
                continue
            ok = True
            for step in range(start, start + rollout_steps - 1):
                if times[step + 1] - times[step] > max_gap:
                    ok = False
                    break
            if ok:
                sequences.append(indices[start:start + rollout_steps])
    if not sequences:
        return np.zeros((0, rollout_steps), dtype=np.int64)
    return np.asarray(sequences, dtype=np.int64)


def run_rollout_epoch(
    model,
    loader,
    arrays_x,
    arrays_y,
    stats_t,
    model_config,
    rollout_weight,
    optimizer=None,
):
    total_loss = 0.0
    total_count = 0
    state_dim = int(model_config["state_dim"])
    action_dim = int(model_config["action_dim"])
    k = int(model_config["k"])
    step_dim = state_dim + action_dim
    use_imu = bool(model_config.get("use_imu", True))
    use_actuator = bool(model_config.get("use_actuator", True))

    for (seq_idx,) in loader:
        seq_idx = seq_idx.to(arrays_x.device)
        x_seq = arrays_x[seq_idx]
        y_seq = arrays_y[seq_idx]
        batch_size, rollout_steps, _ = x_seq.shape
        history = x_seq[:, 0, :].reshape(batch_size, k + 1, step_dim).clone()
        losses = []
        for step in range(rollout_steps):
            model_input = history.reshape(batch_size, -1)
            model_input_n = (model_input - stats_t["x_mean"]) / stats_t["x_std"]
            pred_n = model(model_input_n)
            target_n = (y_seq[:, step, :] - stats_t["y_mean"]) / stats_t["y_std"]
            losses.append(torch.mean((pred_n - target_n) ** 2))

            if step + 1 >= rollout_steps:
                continue

            pred = pred_n * stats_t["y_std"] + stats_t["y_mean"]
            next_state = closed_loop_next_state(
                current_state=history[:, -1, :state_dim],
                pred=pred,
                state_dim=state_dim,
                use_imu=use_imu,
                use_actuator=use_actuator,
            )
            next_action = x_seq[:, step + 1, :].reshape(
                batch_size,
                k + 1,
                step_dim,
            )[:, -1, state_dim:state_dim + action_dim]
            next_step = torch.cat([next_state, next_action], dim=1)
            history = torch.cat([history[:, 1:, :], next_step[:, None, :]], dim=1)

        loss = torch.stack(losses).mean()
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            (rollout_weight * loss).backward()
            optimizer.step()
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
    return total_loss / max(total_count, 1)


def closed_loop_next_state(current_state, pred, state_dim, use_imu, use_actuator):
    next_state = current_state.clone()
    next_state[:, 0:3] = pred[:, 6:9]
    next_state[:, 3:6] = pred[:, 9:12]
    if state_dim >= 9:
        delta_rot = torch_rotvec_to_rotmat(pred[:, 3:6])
        gravity = current_state[:, 6:9]
        next_state[:, 6:9] = torch.bmm(delta_rot.transpose(1, 2), gravity[:, :, None]).squeeze(-1)
    cursor = 9
    if use_imu:
        cursor += 3
    if use_actuator:
        cursor += 4
    return next_state[:, :cursor]


def torch_rotvec_to_rotmat(rotvec):
    batch = rotvec.shape[0]
    theta = torch.linalg.norm(rotvec, dim=1, keepdim=True).clamp_min(1e-8)
    axis = rotvec / theta
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    zeros = torch.zeros_like(x)
    K = torch.stack([
        zeros, -z, y,
        z, zeros, -x,
        -y, x, zeros,
    ], dim=1).reshape(batch, 3, 3)
    eye = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype).expand(batch, 3, 3)
    sin_t = torch.sin(theta).reshape(batch, 1, 1)
    cos_t = torch.cos(theta).reshape(batch, 1, 1)
    return eye + sin_t * K + (1.0 - cos_t) * torch.bmm(K, K)


if __name__ == "__main__":
    main()
