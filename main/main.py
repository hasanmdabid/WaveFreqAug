# This script is written and organized by Md Abid Hasan towards the project WaveFreqAug.
# This is script for training and evaluating time series forecasting models with
# various data augmentation techniques.
# =========================================================================================

import os, sys, pathlib,torch, random
import torch.multiprocessing as mp
from itertools import product
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe in subprocesses
import matplotlib
matplotlib.use("Agg")
from utils.helper import run_combo, load_completed_combos, purge_stale_iterations

_SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
_REPO_ROOT = _SCRIPT_DIR.parent
# The `main.*` absolute imports below need the repo root (parent of this
# script's directory) on sys.path, regardless of how/where this script is
# launched from (terminal, IDE run button, different cwd, etc.).
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Only dataset_configs is needed in the main process; all model/training imports
# happen inside run_combo so each spawned subprocess gets its own clean state.
from utils.dataset_parameter import dataset_configs     # noqa: E402

print("Script directory:", _SCRIPT_DIR)

# ─── VRAM knob ─────────────────────────────────────────────────────────────────
# With many workers sharing one RTX 4090 (24 GB), each run gets a slice of VRAM.
# If you hit OOM, lower this value and restart — completed combos are skipped
# automatically.
BATCH_SIZE_OVERRIDE: int | None = None   # None = use dataset default

# Number of parallel worker processes. DLinear/SCINet/iTransformer/FEDformer are
# all small enough that VRAM is rarely the bottleneck here (~0.5-1 GB/worker) —
# CPU threading is. See CPU_THREADS_PER_WORKER below.
NUM_WORKERS_POOL = 16

# Each worker process defaults to using every CPU core for its BLAS/OMP thread
# pool; with NUM_WORKERS_POOL of them running at once that oversubscribes the
# machine and starves the GPU feed loop.
CPU_THREADS_PER_WORKER = max(1, (os.cpu_count() or NUM_WORKERS_POOL) // NUM_WORKERS_POOL)
# Must be set before numpy/torch import so their BLAS/OMP backends pick it
# up at init — otherwise each worker defaults to using every CPU core and
# NUM_WORKERS_POOL of them thrash each other, starving the GPU feed loop.
os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS_PER_WORKER)
os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS_PER_WORKER)

torch.set_num_threads(CPU_THREADS_PER_WORKER)

# ─── Grid search axes ──────────────────────────────────────────────────────────
MASK_RATES = [0.1, 0.15, 0.2]
LEVELS     = [1, 3, 5]
WAVELETS   = ["db2", "db4", "sym4"]
LAMBDS     = ["U-Shape", "uniform"]
WINDOWS    = [6, 12, 24]


WINDOWS_ILI = [2, 4, 6]

def _windows_for(dataset_name: str) -> list:
    return WINDOWS_ILI if dataset_name == "ILI" else WINDOWS


# ─── Random search ───────────────────────────────────────────────────────────

RANDOM_SEARCH_MODELS = {"DLinear", "SCINet", "iTransformer", "FEDformer", "TiDE", "PatchTST"}
RANDOM_SEARCH_N = 20

# This file's shared lr=0.01 (below) diverges PatchTST to NaN partway through
# its first epoch -- verified empirically on real ETTh1 batches: loss spikes
# into the hundreds/thousands and grad norms hit inf within ~250 steps at
# lr=0.01 (with pre_norm=True and with norm="LayerNorm" too, so it isn't a
# norm-placement fix like iTransformer's), while lr=0.0001 (the learning rate
# PatchTST's own official scripts almost always use) stays stable through a
# full epoch with a healthy, converging loss. Scoped to PatchTST only —
# every other model keeps the shared lr.
PER_MODEL_LR = {"PatchTST": 0.0001}


# ─── Fixed parameters ──────────────────────────────────────────────────────────
AUG_TYPE      = "Wave-Freq"
SAMPLING_RATE = 0.2


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
            use_random_search = model_name in RANDOM_SEARCH_MODELS
            search_mode = "random_search" if use_random_search else "grid_search"
            print(f"\n=== {dataset_name} / {model_name} [{search_mode}] ===")

            results_dir = str(_SCRIPT_DIR / "results" / search_mode / model_name)
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

            completed_combos = load_completed_combos(avg_csv)
            windows = _windows_for(dataset_name)

            # Build the pending combo list for this (dataset, model) pair.
            # DLinear/SCINet: full grid. iTransformer/FEDformer: a fixed,
            # reproducibly-seeded random sample of RANDOM_SEARCH_N points per
            # pred_len instead of the full 162-point grid.
            combos = []
            pending_pairs = set()
            combos_per_pred_len = 0
            for pred_len in config["pred_lens"]:
                grid_points = list(product(MASK_RATES, LEVELS, WAVELETS, LAMBDS, windows))
                if use_random_search:
                    rng = random.Random(f"{dataset_name}_{model_name}_{pred_len}")
                    grid_points = rng.sample(grid_points, min(RANDOM_SEARCH_N, len(grid_points)))
                combos_per_pred_len = len(grid_points)

                for mask_rate, level, wavelet, lambd, window in grid_points:
                    key = (pred_len, mask_rate, level, wavelet, lambd, window)
                    if key in completed_combos:
                        continue
                    pending_pairs.add(key)
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
                        "lr"                 : PER_MODEL_LR.get(model_name, learning_rate),
                        "patience"           : patience,
                        "num_iterations"     : num_iterations,
                        "label_len"          : label_len,
                        "batch_size_override": BATCH_SIZE_OVERRIDE,
                        "sampling_rate"      : SAMPLING_RATE,
                        "aug_type"           : AUG_TYPE,
                        "search_mode"        : search_mode,
                    })

            if not combos:
                print(f"  All combos already completed for {dataset_name}/{model_name}.")
                continue

            removed = purge_stale_iterations(iter_csv, pending_pairs)
            if removed:
                print(f"  Cleared {removed} stale iteration row(s) for combo(s) being rerun.")

            if use_random_search:
                print(f"  Random search: {combos_per_pred_len} of "
                      f"{len(list(product(MASK_RATES, LEVELS, WAVELETS, LAMBDS, windows)))} "
                      f"grid points per pred_len.")
            total = len(config["pred_lens"]) * combos_per_pred_len
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

                    with open(iter_csv, "a", encoding="utf-8") as f:
                        for itr, vl, mae, mse, rse, t in result["iteration_rows"]: # type: ignore
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
    models = ["FEDformer", "iTransformer", "DLinear", "SCINet", "TiDE", "PatchTST"]
    main(models, epochs=20, learning_rate=0.01, patience=5, num_iterations=3, label_len=0)
    print("Main experiment finished.")
    print("You can now check results/ and plots/ for outputs.")
