"""
Ablation study for WaveFreqAug.

Tests two component-removal conditions, each using the best hyperparameters
found in the grid search (extracted from result_analysis/analysis/comparison_*.csv):

  - WaveFreqAug-NoFourier  : Fourier enhancement on the approximation is removed
  - WaveFreqAug-NoMasking  : Adaptive masking on the detail coefficients is removed

Results go to wave_freq/ablation_study/result/{model}/

Run from the repo root:
    cd wave_freq
    python ablation_study/ablation_main.py
"""

import numpy as np
import os
import sys
import pathlib
import torch
import torch.multiprocessing as mp
import matplotlib

matplotlib.use("Agg")

_SCRIPT_DIR = pathlib.Path(__file__).parent.parent.resolve()  # wave_freq/
_REPO_ROOT = _SCRIPT_DIR.parent.resolve()
for p in [str(_SCRIPT_DIR), str(_REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from wave_freq.utils.dataset_parameter import dataset_configs  # noqa: E402

# ─── Workers ──────────────────────────────────────────────────────────────────
NUM_WORKERS_POOL = 4  # lower than main.py; ablation has fewer combos
SAMPLING_RATE = 0.2

# ─── Ablation variants ────────────────────────────────────────────────────────
# "Full" is included as the reference so all three variants are in one CSV.
ABLATION_VARIANTS = [
    "WaveFreqAug-NoFourier",
    "WaveFreqAug-NoMasking",
]

# ─── Best parameters per (dataset, model, pred_len) ──────────────────────────
# Extracted from result_analysis/analysis/comparison_{dataset}_{model}.csv
# by selecting the WaveFreqAug row with lowest MSE for each prediction horizon.
BEST_PARAMS: dict[tuple, dict] = {
    # ETTh1 — DLinear
    ("ETTh1", "DLinear", 96): {
        "mask_rate": 0.20,
        "level": 3,
        "wavelet": "db2",
        "lambd": "U-Shape",
        "window": 24,
    },
    ("ETTh1", "DLinear", 192): {
        "mask_rate": 0.15,
        "level": 3,
        "wavelet": "db4",
        "lambd": "U-Shape",
        "window": 24,
    },
    ("ETTh1", "DLinear", 336): {
        "mask_rate": 0.20,
        "level": 5,
        "wavelet": "sym4",
        "lambd": "U-Shape",
        "window": 24,
    },
    ("ETTh1", "DLinear", 720): {
        "mask_rate": 0.20,
        "level": 5,
        "wavelet": "sym4",
        "lambd": "U-Shape",
        "window": 6,
    },
    # ETTh1 — SCINet
    ("ETTh1", "SCINet", 96): {
        "mask_rate": 0.20,
        "level": 5,
        "wavelet": "sym4",
        "lambd": "uniform",
        "window": 24,
    },
    ("ETTh1", "SCINet", 192): {
        "mask_rate": 0.20,
        "level": 5,
        "wavelet": "sym4",
        "lambd": "U-Shape",
        "window": 24,
    },
    ("ETTh1", "SCINet", 336): {
        "mask_rate": 0.20,
        "level": 5,
        "wavelet": "db2",
        "lambd": "uniform",
        "window": 12,
    },
    ("ETTh1", "SCINet", 720): {
        "mask_rate": 0.20,
        "level": 5,
        "wavelet": "sym4",
        "lambd": "uniform",
        "window": 6,
    },
    # ETTh2 — DLinear
    ("ETTh2", "DLinear", 96): {
        "mask_rate": 0.20,
        "level": 5,
        "wavelet": "db2",
        "lambd": "U-Shape",
        "window": 24,
    },
    ("ETTh2", "DLinear", 192): {
        "mask_rate": 0.15,
        "level": 1,
        "wavelet": "db4",
        "lambd": "U-Shape",
        "window": 6,
    },
    ("ETTh2", "DLinear", 336): {
        "mask_rate": 0.15,
        "level": 5,
        "wavelet": "db4",
        "lambd": "uniform",
        "window": 12,
    },
    ("ETTh2", "DLinear", 720): {
        "mask_rate": 0.20,
        "level": 5,
        "wavelet": "db4",
        "lambd": "uniform",
        "window": 24,
    },
    # ETTh2 — SCINet
    ("ETTh2", "SCINet", 96): {
        "mask_rate": 0.10,
        "level": 3,
        "wavelet": "sym4",
        "lambd": "U-Shape",
        "window": 12,
    },
    ("ETTh2", "SCINet", 192): {
        "mask_rate": 0.15,
        "level": 3,
        "wavelet": "db4",
        "lambd": "U-Shape",
        "window": 12,
    },
    ("ETTh2", "SCINet", 336): {
        "mask_rate": 0.10,
        "level": 5,
        "wavelet": "db4",
        "lambd": "U-Shape",
        "window": 12,
    },
    ("ETTh2", "SCINet", 720): {
        "mask_rate": 0.15,
        "level": 5,
        "wavelet": "db4",
        "lambd": "U-Shape",
        "window": 12,
    },
    # ILI — DLinear
    ("ILI", "DLinear", 24): {
        "mask_rate": 0.20,
        "level": 5,
        "wavelet": "sym4",
        "lambd": "uniform",
        "window": 12,
    },
    ("ILI", "DLinear", 36): {
        "mask_rate": 0.20,
        "level": 5,
        "wavelet": "sym4",
        "lambd": "U-Shape",
        "window": 24,
    },
    ("ILI", "DLinear", 48): {
        "mask_rate": 0.15,
        "level": 5,
        "wavelet": "db4",
        "lambd": "uniform",
        "window": 24,
    },
    ("ILI", "DLinear", 60): {
        "mask_rate": 0.15,
        "level": 5,
        "wavelet": "db4",
        "lambd": "U-Shape",
        "window": 24,
    },
    # ILI — SCINet
    ("ILI", "SCINet", 24): {
        "mask_rate": 0.15,
        "level": 5,
        "wavelet": "db4",
        "lambd": "U-Shape",
        "window": 12,
    },
    ("ILI", "SCINet", 36): {
        "mask_rate": 0.15,
        "level": 5,
        "wavelet": "db2",
        "lambd": "uniform",
        "window": 6,
    },
    ("ILI", "SCINet", 48): {
        "mask_rate": 0.15,
        "level": 3,
        "wavelet": "db2",
        "lambd": "uniform",
        "window": 12,
    },
    ("ILI", "SCINet", 60): {
        "mask_rate": 0.15,
        "level": 3,
        "wavelet": "db4",
        "lambd": "uniform",
        "window": 12,
    },
    # weather — DLinear
    ("weather", "DLinear", 96): {
        "mask_rate": 0.15,
        "level": 5,
        "wavelet": "db2",
        "lambd": "U-Shape",
        "window": 24,
    },
    ("weather", "DLinear", 192): {
        "mask_rate": 0.15,
        "level": 5,
        "wavelet": "db2",
        "lambd": "uniform",
        "window": 24,
    },
    ("weather", "DLinear", 336): {
        "mask_rate": 0.15,
        "level": 5,
        "wavelet": "db2",
        "lambd": "uniform",
        "window": 6,
    },
    ("weather", "DLinear", 720): {
        "mask_rate": 0.10,
        "level": 3,
        "wavelet": "db4",
        "lambd": "U-Shape",
        "window": 6,
    },
    # weather — SCINet
    ("weather", "SCINet", 96): {
        "mask_rate": 0.15,
        "level": 5,
        "wavelet": "sym4",
        "lambd": "U-Shape",
        "window": 6,
    },
    ("weather", "SCINet", 192): {
        "mask_rate": 0.15,
        "level": 3,
        "wavelet": "db2",
        "lambd": "U-Shape",
        "window": 24,
    },
    ("weather", "SCINet", 336): {
        "mask_rate": 0.10,
        "level": 5,
        "wavelet": "sym4",
        "lambd": "uniform",
        "window": 6,
    },
    ("weather", "SCINet", 720): {
        "mask_rate": 0.20,
        "level": 1,
        "wavelet": "db4",
        "lambd": "U-Shape",
        "window": 24,
    },
}


# ─── Worker ───────────────────────────────────────────────────────────────────
def _run_ablation_combo(args: dict):
    """Train + test one (dataset, model, pred_len, aug_variant) inside a subprocess."""
    import gc, os, sys, time, pathlib
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.amp.autocast_mode import autocast
    from torch.amp.grad_scaler import GradScaler
    import matplotlib

    matplotlib.use("Agg")
    from torch.utils.data import DataLoader

    script_dir = pathlib.Path(args["script_dir"])
    repo_root = pathlib.Path(args["repo_root"])
    for p in [str(script_dir), str(repo_root)]:
        if p not in sys.path:
            sys.path.insert(0, p)

    from wave_freq.utils.model import DLinear, SCINet, iTransformer
    from wave_freq.utils.dataloader import TimeSeriesDataset
    from wave_freq.ablation_study.aug_method_ablation import AblationAugmentation

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    dataset_name = args["dataset_name"]
    config = args["config"]
    model_name = args["model_name"]
    pred_len = args["pred_len"]
    aug_variant = args["aug_variant"]
    mask_rate = args["mask_rate"]
    level = args["level"]
    wavelet = args["wavelet"]
    lambd = args["lambd"]
    window = args["window"]
    epochs = args["epochs"]
    lr = args["lr"]
    patience = args["patience"]
    num_iterations = args["num_iterations"]
    label_len = args["label_len"]
    batch_size = args["batch_size"]
    sampling_rate = args["sampling_rate"]
    seq_len = config["seq_len"]

    tag = (
        f"{dataset_name}/{model_name} pred={pred_len} {aug_variant} "
        f"mr={mask_rate} lv={level} wl={wavelet} lm={lambd} win={window}"
    )
    print(f"[pid={os.getpid()}] Starting: {tag}", flush=True)

    checkpoints_dir = (
        script_dir
        / "ablation_study"
        / "checkpoints"
        / (
            f"{dataset_name}_{aug_variant}_{pred_len}_{model_name}"
            f"_{mask_rate}_{level}_{wavelet}_{lambd}_{window}"
        )
    )
    os.makedirs(str(checkpoints_dir), exist_ok=True)

    try:
        ds_kw = dict(
            data_path=config["data_path"],
            data_name=config["data_name"],
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            enc_in=config["enc_in"],
        )
        train_ds = TimeSeriesDataset(**ds_kw, flag="train")
        val_ds = TimeSeriesDataset(**ds_kw, flag="val")
        test_ds = TimeSeriesDataset(**ds_kw, flag="test")
    except Exception as e:
        print(f"[pid={os.getpid()}] Dataset load failed: {e}", flush=True)
        return None

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
        pin_memory=pin,
    )

    def _build_model():
        if model_name == "DLinear":
            return DLinear(
                seq_len, pred_len, enc_in=config["enc_in"], individual=False
            ).to(device)
        if model_name == "SCINet":
            return SCINet(
                input_len=seq_len,
                output_len=pred_len,
                input_dim=config["enc_in"],
                hid_size=1,
                num_stacks=1,
                num_levels=3,
                kernel_size=5,
                dropout=0.2,
            ).to(device)
        if model_name == "iTransformer":
            return iTransformer(
                seq_len=seq_len,
                pred_len=pred_len,
                enc_in=config["enc_in"],
                d_model=512,
                n_heads=8,
                e_layers=4,
                d_ff=2048,
                dropout=0.1,
            ).to(device)
        raise ValueError(f"Unknown model: {model_name}")

    aug_obj = AblationAugmentation()
    if aug_variant == "WaveFreqAug-Full":
        aug_fn = aug_obj.wave_freq_aug_full
    elif aug_variant == "WaveFreqAug-NoFourier":
        aug_fn = aug_obj.wave_freq_aug_no_fourier
    elif aug_variant == "WaveFreqAug-NoMasking":
        aug_fn = aug_obj.wave_freq_aug_no_masking
    else:
        raise ValueError(f"Unknown aug_variant: {aug_variant}")

    aug_kwargs = dict(
        mask_rate=mask_rate,
        wavelet=wavelet,
        level=level,
        lambd=lambd,
        window=window,
        top_k_ratio=sampling_rate,
    )

    # ── EarlyStopping ───────────────────────────────────────────────────────
    class EarlyStopping:
        def __init__(self, patience=7):
            self.patience = patience
            self.counter = 0
            self.best_score = None
            self.early_stop = False
            self.val_loss_min = float("inf")

        def __call__(self, val_loss, model, path):
            score = -val_loss
            if self.best_score is None or score > self.best_score:
                self.best_score = score
                self.val_loss_min = val_loss
                torch.save(model.state_dict(), os.path.join(path, "checkpoint.pth"))
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

    # ── Metrics ─────────────────────────────────────────────────────────────
    def _mae(p, t):
        return float(np.mean(np.abs(p - t)))

    def _mse(p, t):
        return float(np.mean((p - t) ** 2))

    def _rse(p, t):
        return float(
            np.sqrt(np.sum((t - p) ** 2)) / np.sqrt(np.sum((t - t.mean()) ** 2))
        )

    def _validate(model):
        model.eval()
        criterion = nn.SmoothL1Loss()
        losses = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.float().to(device, non_blocking=True)
                by = by.float().to(device, non_blocking=True)
                with autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                    out = model(bx)
                    losses.append(
                        criterion(out[:, -pred_len:, :], by[:, -pred_len:, :]).item()
                    )
        return float(np.mean(losses))

    def _test(model):
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for bx, by in test_loader:
                bx = bx.float().to(device, non_blocking=True)
                by = by.float().to(device, non_blocking=True)
                with autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                    out = model(bx)
                preds.append(out[:, -pred_len:, :].cpu().numpy())
                trues.append(by[:, -pred_len:, :].cpu().numpy())
        preds = np.concatenate(preds, 0).reshape(-1, pred_len, config["enc_in"])
        trues = np.concatenate(trues, 0).reshape(-1, pred_len, config["enc_in"])
        # Metrics are computed in standardised space, matching wave_freq/train_eval.py::test.
        # Inverse-transforming here would report MSE in physical units and make these
        # numbers incomparable with the main experiments.
        return _mae(preds, trues), _mse(preds, trues), _rse(preds, trues)

    mse_list, mae_list, rse_list, val_loss_list, time_list = [], [], [], [], []
    iteration_rows = []

    for itr in range(num_iterations):
        model = None
        itr_start = time.time()
        try:
            model = _build_model()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.SmoothL1Loss()
            scaler = GradScaler("cuda") if device.type == "cuda" else None
            early_stop = EarlyStopping(patience=patience)

            for epoch in range(epochs):
                model.train()
                train_losses = []
                for batch_x, batch_y in train_loader:
                    batch_x_cpu = batch_x.float()
                    batch_y_cpu = batch_y.float()
                    batch_x_dev = batch_x_cpu.to(device, non_blocking=True)
                    batch_y_dev = batch_y_cpu.to(device, non_blocking=True)

                    xy = aug_fn(
                        batch_x_cpu, batch_y_cpu[:, -pred_len:, :], **aug_kwargs
                    )
                    n_sub = int(batch_x_cpu.shape[0] * sampling_rate)
                    idx = torch.randperm(batch_x_cpu.shape[0])[:n_sub]
                    bx2 = xy[idx, :seq_len, :].to(device, non_blocking=True)
                    by2 = xy[idx, seq_len : seq_len + label_len + pred_len, :].to(
                        device, non_blocking=True
                    )

                    optimizer.zero_grad()
                    ctx = (
                        autocast(device_type="cuda")
                        if device.type == "cuda"
                        else autocast(device_type="cpu")
                    )
                    with ctx:
                        out = model(batch_x_dev)
                        loss = (
                            criterion(
                                out[:, -pred_len:, :], batch_y_dev[:, -pred_len:, :]
                            )
                            / 2
                        )
                        out2 = model(bx2)
                        loss = (
                            loss
                            + criterion(out2[:, -pred_len:, :], by2[:, -pred_len:, :])
                            / 2
                        )

                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                    train_losses.append(loss.item())

                val_loss = _validate(model)
                print(
                    f"  [pid={os.getpid()}] epoch={epoch+1} train={np.mean(train_losses):.6f} val={val_loss:.6f} — {tag}",
                    flush=True,
                )

                early_stop(val_loss, model, str(checkpoints_dir))
                if early_stop.early_stop:
                    print(
                        f"  [pid={os.getpid()}] Early stopping at epoch {epoch+1}",
                        flush=True,
                    )
                    break

                # halve LR each epoch (same as main.py)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr * (0.5**epoch)

            ckpt_path = str(checkpoints_dir / "checkpoint.pth")
            if not os.path.exists(ckpt_path):
                print(
                    f"[pid={os.getpid()}] No checkpoint — skipping itr {itr+1}",
                    flush=True,
                )
                continue
            model.load_state_dict(torch.load(ckpt_path, weights_only=True))
            mae, mse, rse = _test(model)
            val_loss_final = early_stop.val_loss_min

        except torch.cuda.OutOfMemoryError as oom:
            print(f"[pid={os.getpid()}] OOM on itr={itr+1} — {tag}: {oom}", flush=True)
            torch.cuda.empty_cache()
            gc.collect()
            break
        except Exception as exc:
            print(
                f"[pid={os.getpid()}] Error on itr={itr+1} — {tag}: {exc}", flush=True
            )
            import traceback

            traceback.print_exc()
            break
        finally:
            if model is not None:
                del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        itr_time = time.time() - itr_start
        mse_list.append(mse)
        mae_list.append(mae)
        rse_list.append(rse)
        val_loss_list.append(val_loss_final)
        time_list.append(itr_time)
        iteration_rows.append((itr + 1, val_loss_final, mae, mse, rse, itr_time))
        print(
            f"[pid={os.getpid()}] itr={itr+1}/{num_iterations} "
            f"MAE={mae:.6f} MSE={mse:.6f} RSE={rse:.6f} t={itr_time:.1f}s — {tag}",
            flush=True,
        )

    if not mse_list:
        return None

    return {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "pred_len": pred_len,
        "aug_variant": aug_variant,
        "mask_rate": mask_rate,
        "level": level,
        "wavelet": wavelet,
        "lambd": lambd,
        "window": window,
        "iteration_rows": iteration_rows,
        "mean_val_loss": float(np.mean(val_loss_list)),
        "mean_mae": float(np.mean(mae_list)),
        "mean_mse": float(np.mean(mse_list)),
        "mean_rse": float(np.mean(rse_list)),
        "std_mae": float(np.std(mae_list)),
        "std_mse": float(np.std(mse_list)),
        "std_rse": float(np.std(rse_list)),
        "mean_time": float(np.mean(time_list)),
    }


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _load_completed(avg_csv: str) -> set:
    completed = set()
    if not os.path.exists(avg_csv):
        return completed
    with open(avg_csv, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if parts[0] in ("dataset", "") or len(parts) < 9:
                continue
            try:
                completed.add((int(parts[2]), parts[3]))  # (pred_len, aug_variant)
            except (ValueError, IndexError):
                continue
    return completed


# ─── Main ─────────────────────────────────────────────────────────────────────
def main(
    models=("DLinear", "SCINet"),
    epochs=50,
    learning_rate=0.01,
    patience=5,
    num_iterations=5,
    label_len=0,
    batch_size=32,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results_root = _SCRIPT_DIR / "ablation_study" / "result"
    ctx = mp.get_context("spawn")

    for dataset_name, config in dataset_configs.items():
        for model_name in models:
            print(f"\n=== Ablation: {dataset_name} / {model_name} ===")

            model_results_dir = results_root / model_name
            os.makedirs(str(model_results_dir), exist_ok=True)

            iter_csv = str(model_results_dir / f"iteration_results_{dataset_name}.csv")
            avg_csv = str(model_results_dir / f"average_results_{dataset_name}.csv")

            if not os.path.exists(iter_csv):
                with open(iter_csv, "w", encoding="utf-8") as f:
                    f.write(
                        "dataset,model,pred_len,aug_variant,mask_rate,level,wavelet,"
                        "lambd,window,iteration,val_loss,mae,mse,rse,exec_time\n"
                    )
            if not os.path.exists(avg_csv):
                with open(avg_csv, "w", encoding="utf-8") as f:
                    f.write(
                        "dataset,model,pred_len,aug_variant,mask_rate,level,wavelet,"
                        "lambd,window,val_loss,mae,mse,rse,mae_std,mse_std,rse_std,exec_time\n"
                    )

            completed = _load_completed(avg_csv)

            combos = []
            for pred_len in config["pred_lens"]:
                key = (dataset_name, model_name, pred_len)
                if key not in BEST_PARAMS:
                    print(f"  WARNING: no best params for {key} — skipping", flush=True)
                    continue
                params = BEST_PARAMS[key]
                for av in ABLATION_VARIANTS:
                    if (pred_len, av) in completed:
                        print(
                            f"  Skipping already-completed: pred={pred_len} {av}",
                            flush=True,
                        )
                        continue
                    combos.append(
                        {
                            "script_dir": str(_SCRIPT_DIR),
                            "repo_root": str(_REPO_ROOT),
                            "dataset_name": dataset_name,
                            "config": config,
                            "model_name": model_name,
                            "pred_len": pred_len,
                            "aug_variant": av,
                            "mask_rate": params["mask_rate"],
                            "level": params["level"],
                            "wavelet": params["wavelet"],
                            "lambd": params["lambd"],
                            "window": params["window"],
                            "epochs": epochs,
                            "lr": learning_rate,
                            "patience": patience,
                            "num_iterations": num_iterations,
                            "label_len": label_len,
                            "batch_size": batch_size,
                            "sampling_rate": SAMPLING_RATE,
                        }
                    )

            if not combos:
                print(
                    f"  All ablation combos already done for {dataset_name}/{model_name}."
                )
                continue

            print(
                f"  {len(combos)} ablation combos to run with {NUM_WORKERS_POOL} workers."
            )

            with ctx.Pool(processes=NUM_WORKERS_POOL, maxtasksperchild=1) as pool:
                for result in pool.imap_unordered(_run_ablation_combo, combos):
                    if result is None:
                        continue

                    with open(iter_csv, "a", encoding="utf-8") as f:
                        for itr, vl, mae, mse, rse, t in result["iteration_rows"]:
                            f.write(
                                f"{result['dataset_name']},{result['model_name']},"
                                f"{result['pred_len']},{result['aug_variant']},"
                                f"{result['mask_rate']},{result['level']},"
                                f"{result['wavelet']},{result['lambd']},{result['window']},"
                                f"{itr},{vl:.6f},{mae:.6f},{mse:.6f},{rse:.6f},{t:.2f}\n"
                            )

                    with open(avg_csv, "a", encoding="utf-8") as f:
                        f.write(
                            f"{result['dataset_name']},{result['model_name']},"
                            f"{result['pred_len']},{result['aug_variant']},"
                            f"{result['mask_rate']},{result['level']},"
                            f"{result['wavelet']},{result['lambd']},{result['window']},"
                            f"{result['mean_val_loss']:.6f},{result['mean_mae']:.6f},"
                            f"{result['mean_mse']:.6f},{result['mean_rse']:.6f},"
                            f"{result['std_mae']:.6f},{result['std_mse']:.6f},"
                            f"{result['std_rse']:.6f},{result['mean_time']:.2f}\n"
                        )

                    print(
                        f"  [Saved] pred={result['pred_len']} {result['aug_variant']} "
                        f"MAE={result['mean_mae']:.6f} MSE={result['mean_mse']:.6f} "
                        f"t={result['mean_time']:.1f}s",
                        flush=True,
                    )

    print("\nAblation study complete.")


if __name__ == "__main__":

    main(
        models=["DLinear", "SCINet"],
        epochs=30,
        learning_rate=0.01,
        patience=10,
        num_iterations=3,
        label_len=0,
        batch_size=32,
    )
