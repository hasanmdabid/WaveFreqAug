import gc
import os
import pathlib
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
# ──────────────────────────────────────────────────────────────────────────────
# Worker — must be a module-level function to be picklable by multiprocessing
# ──────────────────────────────────────────────────────────────────────────────
def run_combo(args: dict):
    """
    Train + test one grid-search combination inside a spawned subprocess.
    Returns a result dict on success, or None if all iterations failed.
    """
    script_dir = pathlib.Path(args["script_dir"])
    repo_root = script_dir.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from main.utils.model import DLinear, SCINet, iTransformer, FEDformer, TiDE, PatchTST
    from main.utils.train_eval import train, test
    from main.utils.dataloader import TimeSeriesDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    dataset_name = args["dataset_name"]
    config = args["config"]
    model_name = args["model_name"]
    pred_len = args["pred_len"]
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
    batch_size = args["batch_size_override"] or config["batch_size"]
    sampling_rate = args["sampling_rate"]
    aug_type = args["aug_type"]
    search_mode = args["search_mode"]
    seq_len = config["seq_len"]

    tag = (
        f"[{search_mode}] {dataset_name}/{model_name} pred={pred_len} mr={mask_rate} "
        f"lv={level} wl={wavelet} lm={lambd} win={window}"
    )
    print(f"[pid={os.getpid()}] Starting: {tag}", flush=True)

    checkpoints_root = script_dir / "checkpoints" / search_mode
    plots_root = script_dir / "plots" / search_mode
    checkpoint_dir = str(
        checkpoints_root
        / (
            f"{dataset_name}_{aug_type}_{pred_len}_{model_name}"
            f"_{mask_rate}_{level}_{wavelet}_{lambd}_{window}"
        )
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(str(plots_root), exist_ok=True)

    # Load datasets — num_workers=0 avoids spawning grandchild processes
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
        print(f"[pid={os.getpid()}] Dataset load failed for {tag}: {e}", flush=True)
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
                e_layers=2,
                d_ff=512,
                dropout=0.1,
            ).to(device)
        if model_name == "FEDformer":
            return FEDformer(
                seq_len=seq_len,
                pred_len=pred_len,
                enc_in=config["enc_in"],
            ).to(device)
        if model_name == "TiDE":
            return TiDE(
                seq_len=seq_len,
                pred_len=pred_len,
                enc_in=config["enc_in"],
                d_model=256,
                e_layers=2,
                d_layers=2,
                d_ff=256,
                dropout=0.1,
            ).to(device)
        if model_name == "PatchTST":
            return PatchTST(
                seq_len=seq_len,
                pred_len=pred_len,
                enc_in=config["enc_in"],
                e_layers=3,
                n_heads=16,
                d_model=128,
                d_ff=256,
                dropout=0.2,
                fc_dropout=0.2,
                patch_len=16,
                stride=8,
            ).to(device)
        raise ValueError(f"Unknown model: {model_name}")

    mse_list, mae_list, rse_list, val_loss_list, time_list = [], [], [], [], []
    iteration_rows = []

    for itr in range(num_iterations):
        model = None
        itr_start = time.time()
        try:
            model = _build_model()
            val_loss = train(
                model,
                train_loader,
                val_loader,
                device,
                aug_type,
                seq_len,
                label_len,
                pred_len,
                dataset_name=dataset_name,
                mask_rate=mask_rate,
                wavelet=wavelet,
                level=level,
                sampling_rate=sampling_rate,
                lambd=lambd,
                window=window,
                epochs=epochs,
                lr=lr,
                patience=patience,
                checkpoint_dir=checkpoint_dir,
            )
            ckpt = os.path.join(checkpoint_dir, "checkpoint.pth")
            if not os.path.exists(ckpt):
                print(
                    f"[pid={os.getpid()}] No checkpoint at {ckpt}, skipping itr {itr+1}",
                    flush=True,
                )
                continue
            model.load_state_dict(torch.load(ckpt, weights_only=True))
            mae, mse, rse = test(model, test_loader, device, train_ds.scaler, pred_len)

        except torch.cuda.OutOfMemoryError as oom:
            # Non-fatal: return partial results from iterations that already succeeded
            print(
                f"[pid={os.getpid()}] GPU OOM on itr={itr+1} — {tag}: {oom}\n"
                f"  Tip: set BATCH_SIZE_OVERRIDE to a lower value (current batch_size={batch_size}).",
                flush=True,
            )
            torch.cuda.empty_cache()
            gc.collect()
            break
        except Exception as exc:
            print(
                f"[pid={os.getpid()}] Error on itr={itr+1} — {tag}: {exc}", flush=True
            )
            break
        finally:
            if model is not None:
                del model
            torch.cuda.empty_cache()

        itr_time = time.time() - itr_start
        mse_list.append(mse)
        mae_list.append(mae)
        rse_list.append(rse)
        val_loss_list.append(val_loss)
        time_list.append(itr_time)
        iteration_rows.append((itr + 1, val_loss, mae, mse, rse, itr_time))
        print(
            f"[pid={os.getpid()}] itr={itr+1}/{num_iterations} val={val_loss:.6f} "
            f"MAE={mae:.6f} MSE={mse:.6f} RSE={rse:.6f} t={itr_time:.1f}s — {tag}",
            flush=True,
        )

    if not mse_list:
        return None

    # Plot using best checkpoint (failures here are non-fatal)
    try:
        plot_model = _build_model()
        plot_model.load_state_dict(
            torch.load(
                os.path.join(checkpoint_dir, "checkpoint.pth"), weights_only=True
            )
        )
        plot_model.eval()
        with torch.no_grad():
            bx, by = next(iter(test_loader))
            bx = bx.float().to(device)
            out = plot_model(bx)[:, -pred_len:, :].cpu().numpy()
            by = by[:, -pred_len:, :].cpu().numpy()
            out = train_ds.scaler.inverse_transform(out[0])
            by = train_ds.scaler.inverse_transform(by[0])
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(by[:, 0], label="Ground Truth")
        plt.plot(out[:, 0], label=f"Prediction ({aug_type}, {model_name})")
        plt.legend()
        plt.savefig(
            str(
                plots_root
                / (
                    f"prediction_{dataset_name}_{aug_type}_{pred_len}"
                    f"_{model_name}_{mask_rate}_{level}_{wavelet}_{lambd}_{window}.png"
                )
            )
        )
        plt.close()
        del plot_model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"[pid={os.getpid()}] Plot skipped (non-fatal): {e}", flush=True)

    return {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "pred_len": pred_len,
        "aug_type": aug_type,
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


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def load_completed_combos(avg_csv: str) -> set:
    """Parse average_results CSV and return set of completed (pred_len, mask_rate, level, wavelet, lambd, window)."""
    completed: set = set()
    if not os.path.exists(avg_csv):
        return completed
    with open(avg_csv, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if parts[0] == "dataset" or len(parts) < 9:
                continue
            try:
                if len(parts) >= 17:  # new schema with lambd, window, exec_time
                    completed.add(
                        (
                            int(parts[2]),
                            float(parts[4]),
                            int(parts[5]),
                            parts[6],
                            parts[7],
                            int(parts[8]),
                        )
                    )
                else:  # old schema — assume U-Shape / window=12
                    completed.add(
                        (
                            int(parts[2]),
                            float(parts[4]),
                            int(parts[5]),
                            parts[6],
                            "U-Shape",
                            12,
                        )
                    )
            except (ValueError, IndexError):
                continue
    return completed


def purge_stale_iterations(iter_csv: str, pairs: set) -> int:
    """Remove any existing iteration rows for the given (pred_len, mask_rate,
    level, wavelet, lambd, window) combos from iter_csv. Used when a combo is
    about to be rerun because it has no average row yet — clears out any
    orphaned iteration data (e.g. left over from a run interrupted between
    writing the iteration rows and the average row) so it doesn't mix with the
    fresh rerun. Returns the number of rows removed."""
    if not pairs or not os.path.exists(iter_csv):
        return 0
    with open(iter_csv, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        return 0
    header, rows = lines[0], lines[1:]
    kept = []
    removed = 0
    for line in rows:
        parts = line.strip().split(",")
        key = None
        if len(parts) >= 9:
            try:
                if len(parts) >= 15:  # new schema with lambd, window, exec_time
                    key = (
                        int(parts[2]),
                        float(parts[4]),
                        int(parts[5]),
                        parts[6],
                        parts[7],
                        int(parts[8]),
                    )
                else:  # old schema — assume U-Shape / window=12
                    key = (
                        int(parts[2]),
                        float(parts[4]),
                        int(parts[5]),
                        parts[6],
                        "U-Shape",
                        12,
                    )
            except (ValueError, IndexError):
                key = None
        if key in pairs:
            removed += 1
            continue
        kept.append(line)
    if removed:
        with open(iter_csv, "w", encoding="utf-8") as f:
            f.write(header)
            f.writelines(kept)
    return removed
