import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    return data["x"].astype(np.float32), data["y"].astype(np.float32)


def normalize(x, mean, std):
    return (x - mean) / std


def main():
    parser = argparse.ArgumentParser(description="Train a history-stacked MLP drone dynamics model.")
    parser.add_argument("--data-dir", default=str(ROOT / "logs" / "processed"))
    parser.add_argument("--checkpoint-dir", default=str(ROOT / "checkpoints"))
    parser.add_argument("--config-dir", default=str(ROOT / "configs"))
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from models.mlp_dynamics import build_model
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = Path(args.data_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    config_dir = Path(args.config_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train = load_npz(data_dir / "train.npz")
    x_val, y_val = load_npz(data_dir / "val.npz")
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
        "input_dim": int(stats["input_dim"]),
        "output_dim": int(stats["output_dim"]),
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "k": int(stats["k"]),
        "state_dim": int(stats["state_dim"]),
        "action_dim": int(stats["action_dim"]),
        "use_actuator": bool(stats["use_actuator"]),
    }
    model = build_model(config).to(args.device)
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

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = run_epoch(model, train_loader, loss_fn, args.device, optimizer)
        model.eval()
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, loss_fn, args.device, optimizer=None)

        curves.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}", flush=True)

        if val_loss < best_val:
            best_val = val_loss
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
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss"])
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


if __name__ == "__main__":
    main()
