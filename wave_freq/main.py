import os
import sys
import pathlib
import torch
import torch.multiprocessing as mp
from itertools import product
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe in subprocesses

_SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# Only dataset_configs is needed in the main process; all model/training imports
# happen inside _run_combo so each spawned subprocess gets its own clean state.
from wave_freq.utils.dataset_parameter import dataset_configs     # noqa: E402

print("Script directory:", _SCRIPT_DIR)

# ─── VRAM knob ─────────────────────────────────────────────────────────────────
# With 8 workers sharing one RTX 4090 (24 GB), each run gets ~3 GB.
# If you hit OOM, lower this value (e.g. 16 or 8) and restart — completed
# combos are skipped automatically.
BATCH_SIZE_OVERRIDE: int | None = None   # None = use dataset default

# Number of parallel worker processes
NUM_WORKERS_POOL = 8

# ─── Grid search axes ──────────────────────────────────────────────────────────
MASK_RATES = [0.1, 0.15, 0.2]
LEVELS     = [1, 3, 5]
WAVELETS   = ["db2", "db4", "sym4"]
LAMBDS     = ["U-Shape", "uniform"]
WINDOWS    = [6, 12, 24]

# ─── Fixed parameters ──────────────────────────────────────────────────────────
AUG_TYPE      = "Wave-Freq"
SAMPLING_RATE = 0.2


# ──────────────────────────────────────────────────────────────────────────────
# Worker — must be a module-level function to be picklable by multiprocessing
# ──────────────────────────────────────────────────────────────────────────────
def _run_combo(args: dict):
    """
    Train + test one grid-search combination inside a spawned subprocess.
    Returns a result dict on success, or None if all iterations failed.
    """
    import gc, os, sys, time, pathlib
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    script_dir = pathlib.Path(args["script_dir"])
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    from wave_freq.utils.model import DLinear, SCINet, iTransformer
    from wave_freq.utils.train_eval import train, test
    from wave_freq.utils.dataloader import TimeSeriesDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    dataset_name   = args["dataset_name"]
    config         = args["config"]
    model_name     = args["model_name"]
    pred_len       = args["pred_len"]
    mask_rate      = args["mask_rate"]
    level          = args["level"]
    wavelet        = args["wavelet"]
    lambd          = args["lambd"]
    window         = args["window"]
    epochs         = args["epochs"]
    lr             = args["lr"]
    patience       = args["patience"]
    num_iterations = args["num_iterations"]
    label_len      = args["label_len"]
    batch_size     = args["batch_size_override"] or config["batch_size"]
    sampling_rate  = args["sampling_rate"]
    aug_type       = args["aug_type"]
    seq_len        = config["seq_len"]

    tag = (f"{dataset_name}/{model_name} pred={pred_len} mr={mask_rate} "
           f"lv={level} wl={wavelet} lm={lambd} win={window}")
    print(f"[pid={os.getpid()}] Starting: {tag}", flush=True)

    checkpoints_root = script_dir / "checkpoints"
    plots_root       = script_dir / "plots"
    checkpoint_dir   = str(checkpoints_root / (
        f"{dataset_name}_{aug_type}_{pred_len}_{model_name}"
        f"_{mask_rate}_{level}_{wavelet}_{lambd}_{window}"
    ))
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(str(plots_root), exist_ok=True)

    # Load datasets — num_workers=0 avoids spawning grandchild processes
    try:
        ds_kw = dict(
            data_path=config["data_path"], data_name=config["data_name"],
            seq_len=seq_len, label_len=label_len, pred_len=pred_len,
            enc_in=config["enc_in"],
        )
        train_ds = TimeSeriesDataset(**ds_kw, flag="train")
        val_ds   = TimeSeriesDataset(**ds_kw, flag="val")
        test_ds  = TimeSeriesDataset(**ds_kw, flag="test")
    except Exception as e:
        print(f"[pid={os.getpid()}] Dataset load failed for {tag}: {e}", flush=True)
        return None

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True, num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0, pin_memory=pin)

    def _build_model():
        if model_name == "DLinear":
            return DLinear(seq_len, pred_len, enc_in=config["enc_in"], individual=False).to(device)
        if model_name == "SCINet":
            return SCINet(
                input_len=seq_len, output_len=pred_len, input_dim=config["enc_in"],
                hid_size=1, num_stacks=1, num_levels=3, kernel_size=5, dropout=0.2,
            ).to(device)
        if model_name == "iTransformer":
            return iTransformer(
                seq_len=seq_len, pred_len=pred_len, enc_in=config["enc_in"],
                d_model=512, n_heads=8, e_layers=4, d_ff=2048, dropout=0.1,
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
                model, train_loader, val_loader, device, aug_type,
                seq_len, label_len, pred_len,
                dataset_name=dataset_name, mask_rate=mask_rate, wavelet=wavelet,
                level=level, sampling_rate=sampling_rate, lambd=lambd, window=window,
                epochs=epochs, lr=lr, patience=patience, checkpoint_dir=checkpoint_dir,
            )
            ckpt = os.path.join(checkpoint_dir, "checkpoint.pth")
            if not os.path.exists(ckpt):
                print(f"[pid={os.getpid()}] No checkpoint at {ckpt}, skipping itr {itr+1}", flush=True)
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
            print(f"[pid={os.getpid()}] Error on itr={itr+1} — {tag}: {exc}", flush=True)
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
            torch.load(os.path.join(checkpoint_dir, "checkpoint.pth"), weights_only=True)
        )
        plot_model.eval()
        with torch.no_grad():
            bx, by = next(iter(test_loader))
            bx  = bx.float().to(device)
            out = plot_model(bx)[:, -pred_len:, :].cpu().numpy()
            by  = by[:, -pred_len:, :].cpu().numpy()
            out = train_ds.scaler.inverse_transform(out[0])
            by  = train_ds.scaler.inverse_transform(by[0])
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(by[:, 0], label="Ground Truth")
        plt.plot(out[:, 0], label=f"Prediction ({aug_type}, {model_name})")
        plt.legend()
        plt.savefig(str(plots_root / (
            f"prediction_{dataset_name}_{aug_type}_{pred_len}"
            f"_{model_name}_{mask_rate}_{level}_{wavelet}_{lambd}_{window}.png"
        )))
        plt.close()
        del plot_model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"[pid={os.getpid()}] Plot skipped (non-fatal): {e}", flush=True)

    return {
        "dataset_name"  : dataset_name,
        "model_name"    : model_name,
        "pred_len"      : pred_len,
        "aug_type"      : aug_type,
        "mask_rate"     : mask_rate,
        "level"         : level,
        "wavelet"       : wavelet,
        "lambd"         : lambd,
        "window"        : window,
        "iteration_rows": iteration_rows,
        "mean_val_loss" : float(np.mean(val_loss_list)),
        "mean_mae"      : float(np.mean(mae_list)),
        "mean_mse"      : float(np.mean(mse_list)),
        "mean_rse"      : float(np.mean(rse_list)),
        "std_mae"       : float(np.std(mae_list)),
        "std_mse"       : float(np.std(mse_list)),
        "std_rse"       : float(np.std(rse_list)),
        "mean_time"     : float(np.mean(time_list)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _load_completed_combos(avg_csv: str) -> set:
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
                if len(parts) >= 17:   # new schema with lambd, window, exec_time
                    completed.add((int(parts[2]), float(parts[4]), int(parts[5]),
                                   parts[6], parts[7], int(parts[8])))
                else:                   # old schema — assume U-Shape / window=12
                    completed.add((int(parts[2]), float(parts[4]), int(parts[5]),
                                   parts[6], "U-Shape", 12))
            except (ValueError, IndexError):
                continue
    return completed


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main(models, epochs, learning_rate, patience, num_iterations, label_len):
    print(f"Script dir : {_SCRIPT_DIR}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    os.makedirs(str(_SCRIPT_DIR / "checkpoints"), exist_ok=True)
    os.makedirs(str(_SCRIPT_DIR / "plots"), exist_ok=True)

    # Use 'spawn' — required for CUDA; 'fork' corrupts GPU contexts
    ctx = mp.get_context("spawn")

    for dataset_name, config in dataset_configs.items():
        for model_name in models:
            print(f"\n=== {dataset_name} / {model_name} ===")

            results_dir = str(_SCRIPT_DIR / "results" / model_name)
            os.makedirs(results_dir, exist_ok=True)
            iter_csv = os.path.join(results_dir, f"iteration_results_{dataset_name}.csv")
            avg_csv  = os.path.join(results_dir, f"average_results_{dataset_name}.csv")

            if not os.path.exists(iter_csv):
                with open(iter_csv, "w", encoding="utf-8") as f:
                    f.write(
                        "dataset,model,pred_len,aug_type,mask_rate,level,wavelet,"
                        "lambd,window,iteration,val_loss,mae,mse,rse,exec_time\n"
                    )
            if not os.path.exists(avg_csv):
                with open(avg_csv, "w", encoding="utf-8") as f:
                    f.write(
                        "dataset,model,pred_len,aug_type,mask_rate,level,wavelet,"
                        "lambd,window,val_loss,mae,mse,rse,mae_std,mse_std,rse_std,exec_time\n"
                    )

            completed_combos = _load_completed_combos(avg_csv)

            # Build the pending combo list for this (dataset, model) pair
            combos = []
            for pred_len in config["pred_lens"]:
                for mask_rate, level, wavelet, lambd, window in product(
                    MASK_RATES, LEVELS, WAVELETS, LAMBDS, WINDOWS
                ):
                    if (pred_len, mask_rate, level, wavelet, lambd, window) in completed_combos:
                        continue
                    combos.append({
                        "script_dir"         : str(_SCRIPT_DIR),
                        "dataset_name"       : dataset_name,
                        "config"             : config,
                        "model_name"         : model_name,
                        "pred_len"           : pred_len,
                        "mask_rate"          : mask_rate,
                        "level"              : level,
                        "wavelet"            : wavelet,
                        "lambd"              : lambd,
                        "window"             : window,
                        "epochs"             : epochs,
                        "lr"                 : learning_rate,
                        "patience"           : patience,
                        "num_iterations"     : num_iterations,
                        "label_len"          : label_len,
                        "batch_size_override": BATCH_SIZE_OVERRIDE,
                        "sampling_rate"      : SAMPLING_RATE,
                        "aug_type"           : AUG_TYPE,
                    })

            if not combos:
                print(f"  All combos already completed for {dataset_name}/{model_name}.")
                continue

            combos_per_pred_len = len(list(product(MASK_RATES, LEVELS, WAVELETS, LAMBDS, WINDOWS)))
            total = len(config["pred_lens"]) * combos_per_pred_len
            print(
                f"  {len(combos)} combos pending, {total - len(combos)} already completed.\n"
                f"  Launching {NUM_WORKERS_POOL} parallel workers "
                f"(batch_size={BATCH_SIZE_OVERRIDE or 'dataset default'})."
            )

            # imap_unordered dispatches combos lazily; the main process writes CSV
            # as each result arrives — no concurrent file access.
            with ctx.Pool(processes=NUM_WORKERS_POOL, maxtasksperchild=1) as pool:
                for result in pool.imap_unordered(_run_combo, combos):
                    if result is None:
                        continue

                    with open(iter_csv, "a", encoding="utf-8") as f:
                        for itr, vl, mae, mse, rse, t in result["iteration_rows"]:
                            f.write(
                                f"{result['dataset_name']},{result['model_name']},"
                                f"{result['pred_len']},{result['aug_type']},"
                                f"{result['mask_rate']},{result['level']},"
                                f"{result['wavelet']},{result['lambd']},{result['window']},"
                                f"{itr},{vl:.6f},{mae:.6f},{mse:.6f},{rse:.6f},{t:.2f}\n"
                            )

                    with open(avg_csv, "a", encoding="utf-8") as f:
                        f.write(
                            f"{result['dataset_name']},{result['model_name']},"
                            f"{result['pred_len']},{result['aug_type']},"
                            f"{result['mask_rate']},{result['level']},"
                            f"{result['wavelet']},{result['lambd']},{result['window']},"
                            f"{result['mean_val_loss']:.6f},{result['mean_mae']:.6f},"
                            f"{result['mean_mse']:.6f},{result['mean_rse']:.6f},"
                            f"{result['std_mae']:.6f},{result['std_mse']:.6f},"
                            f"{result['std_rse']:.6f},{result['mean_time']:.2f}\n"
                        )

                    print(
                        f"  [Saved] pred={result['pred_len']} mr={result['mask_rate']} "
                        f"lv={result['level']} wl={result['wavelet']} "
                        f"lm={result['lambd']} win={result['window']} → "
                        f"MAE={result['mean_mae']:.6f} MSE={result['mean_mse']:.6f} "
                        f"t={result['mean_time']:.1f}s"
                    )

    print("\nAll experiments completed.")


if __name__ == "__main__":
    print("Starting main experiment...")
    models = ["DLinear", "SCINet"]
    main(models, epochs=50, learning_rate=0.01, patience=15, num_iterations=5, label_len=0)
    print("Main experiment finished.")
    print("You can now check results/ and plots/ for outputs.")
