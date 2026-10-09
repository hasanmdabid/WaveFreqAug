# This script is written and organized by Md Abid Hasan towards the project WaveFreqAug.
# This is script for training and evaluating time series forecasting models with
# various data augmentation techniques.
# =========================================================================================


import sys
import pathlib

_SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
_REPO_ROOT = _SCRIPT_DIR.parent
# The `baseline.*` absolute imports below need the repo root (parent of this
# script's directory) on sys.path, regardless of how/where this script is
# launched from (terminal, IDE run button, different cwd, etc.).
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import os
import torch
import torch.multiprocessing as mp
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe in subprocesses

# Only dataset_configs is needed in the main process; all model/training imports
# happen inside run_combo so each spawned subprocess gets its own clean state.
from utils.dataset_parameter import dataset_configs   # noqa: E402

print("Script directory:", _SCRIPT_DIR)

# ─── VRAM knob ─────────────────────────────────────────────────────────────────
# With many workers sharing one GPU, each run gets a slice of VRAM.
# If you hit OOM, lower this value and restart — completed combos are skipped
# automatically.
BATCH_SIZE_OVERRIDE: int | None = None   # None = use dataset default

# Number of parallel worker processes. DLinear/SCINet/iTransformer/FEDformer are
# all small enough that VRAM is rarely the bottleneck here (~0.5-1 GB/worker) —
# CPU threading is. See CPU_THREADS_PER_WORKER below.
NUM_WORKERS_POOL = 28

# Each worker process defaults to using every CPU core for its BLAS/OMP thread
# pool; with NUM_WORKERS_POOL of them running at once that oversubscribes the
# machine and starves the GPU feed loop (symptom: near-0% GPU util despite
# multiple workers "running"). Cap each worker to a fair share of the cores instead.
CPU_THREADS_PER_WORKER = max(1, (os.cpu_count() or NUM_WORKERS_POOL) // NUM_WORKERS_POOL)

# The shared lr=0.01 (passed to main() at the bottom of this file) diverges
# PatchTST to NaN partway through its first epoch -- verified empirically on
# real ETTh1 batches: loss spikes into the hundreds/thousands and grad norms
# hit inf within ~250 steps at lr=0.01 (with pre_norm=True and with
# norm="LayerNorm" too, so it isn't a norm-placement fix like iTransformer's),
# while lr=0.0001 (the learning rate PatchTST's own official scripts almost
# always use) stays stable through a full epoch with a healthy, converging
# loss. Scoped to PatchTST only — every other model keeps the shared lr.
PER_MODEL_LR = {"PatchTST": 0.0001}

from utils.helper import run_combo, load_completed_combos, purge_stale_iterations

# Must be set before numpy/torch import so their BLAS/OMP backends pick it
# up at init — otherwise each worker defaults to using every CPU core and
# NUM_WORKERS_POOL of them thrash each other, starving the GPU feed loop.
os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS_PER_WORKER)
os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS_PER_WORKER)
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

torch.set_num_threads(CPU_THREADS_PER_WORKER)

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main(models, epochs, learning_rate, patience, num_iterations, label_len):
    print("Starting main experiment...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    os.makedirs(str(_SCRIPT_DIR / "checkpoints"), exist_ok=True)
    os.makedirs(str(_SCRIPT_DIR / "plots"), exist_ok=True)

    results_root = _SCRIPT_DIR / "results"

    # Use 'spawn' — required for CUDA; 'fork' corrupts GPU contexts
    ctx = mp.get_context("spawn")

    for dataset_name, config in dataset_configs.items():
        for model_name in models:
            print(f"\n=== Processing dataset: {dataset_name} with model: {model_name} ===")

            # Per-model results directory. Open in append mode and only write the
            # header when the file is new, so previously saved results are never erased.
            model_results_dir = str(results_root / model_name)
            os.makedirs(model_results_dir, exist_ok=True)

            iteration_csv_path = os.path.join(model_results_dir, f"iteration_results_{dataset_name}.csv")
            if not os.path.exists(iteration_csv_path):
                with open(iteration_csv_path, "w") as f_iter:
                    f_iter.write(
                        "dataset,model,pred_len,aug_type,iteration,val_loss,mae,mse,rse,exec_time\n"
                    )

            average_csv_path = os.path.join(model_results_dir, f"average_results_{dataset_name}.csv")
            if not os.path.exists(average_csv_path):
                with open(average_csv_path, "w") as f_avg:
                    f_avg.write(
                        "dataset,model,pred_len,aug_type,val_loss,mae,mse,rse,mae_std,mse_std,rse_std,exec_time\n"
                    )

            # Build set of (pred_len, aug_type) pairs that already have a
            # recorded average result, so re-running does not duplicate or
            # overwrite completed experiments.
            completed_combos = load_completed_combos(average_csv_path)

            combos = []
            pending_pairs = set()
            for pred_len in config["pred_lens"]:
                for aug_type in config["aug_types"]:
                    if (pred_len, aug_type) in completed_combos:
                        continue
                    pending_pairs.add((pred_len, aug_type))
                    combos.append({
                        "script_dir"         : str(_SCRIPT_DIR),
                        "dataset_name"       : dataset_name,
                        "config"             : config,
                        "model_name"         : model_name,
                        "pred_len"           : pred_len,
                        "aug_type"           : aug_type,
                        "params"             : config["aug_params"][pred_len][aug_type],
                        "epochs"             : epochs,
                        "lr"                 : PER_MODEL_LR.get(model_name, learning_rate),
                        "patience"           : patience,
                        "num_iterations"     : num_iterations,
                        "label_len"          : label_len,
                        "batch_size_override": BATCH_SIZE_OVERRIDE,
                    })

            if not combos:
                print(f"  All combos already completed for {dataset_name}/{model_name}.")
                continue

            removed = purge_stale_iterations(iteration_csv_path, pending_pairs)
            if removed:
                print(f"  Cleared {removed} stale iteration row(s) for combo(s) being rerun.")

            total = len(config["pred_lens"]) * len(config["aug_types"])
            print(
                f"  {len(combos)} combos pending, {total - len(combos)} already completed.\n"
                f"  Launching {NUM_WORKERS_POOL} parallel workers "
                f"(batch_size={BATCH_SIZE_OVERRIDE or 'dataset default'})."
            )

            # imap_unordered dispatches combos lazily; the main process writes CSV
            # as each result arrives — no concurrent file access.
            with ctx.Pool(processes=NUM_WORKERS_POOL, maxtasksperchild=1) as pool:
                for result in pool.imap_unordered(run_combo, combos):
                    if result is None:
                        continue

                    with open(iteration_csv_path, "a") as f_iter:
                        for itr, vl, mae, mse, rse, t in result["iteration_rows"]:
                            f_iter.write(
                                f"{result['dataset_name']},{result['model_name']},"
                                f"{result['pred_len']},{result['aug_type']},{itr},"
                                f"{vl:.6f},{mae:.6f},{mse:.6f},{rse:.6f},{t:.2f}\n"
                            )

                    with open(average_csv_path, "a") as f_avg:
                        f_avg.write(
                            f"{result['dataset_name']},{result['model_name']},"
                            f"{result['pred_len']},{result['aug_type']},"
                            f"{result['mean_val_loss']:.6f},{result['mean_mae']:.6f},"
                            f"{result['mean_mse']:.6f},{result['mean_rse']:.6f},"
                            f"{result['std_mae']:.6f},{result['std_mse']:.6f},"
                            f"{result['std_rse']:.6f},{result['mean_time']:.2f}\n"
                        )

                    print(
                        f"  [Saved] {result['dataset_name']}/{result['model_name']} "
                        f"pred={result['pred_len']} aug={result['aug_type']} → "
                        f"MAE={result['mean_mae']:.6f} MSE={result['mean_mse']:.6f} "
                        f"t={result['mean_time']:.1f}s"
                    )

    print("\nAll experiments completed.")


if __name__ == "__main__":
    models = ["DLinear", "SCINet", "FEDformer", "iTransformer", "TiDE", "PatchTST"]
    main(models=models, epochs=20, learning_rate=0.01, patience=5, num_iterations=3, label_len=0)
    print("Main experiment finished.")
    print("You can now check the results and plots in the respective directories.")
