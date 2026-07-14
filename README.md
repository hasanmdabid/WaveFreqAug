# WaveFreqAug: Wavelet Guided Frequency Augmentation for Robust Time Series Forecasting

## Overview

**WaveFreqAug** is a data augmentation algorithm for multivariate time series forecasting. It combines three signal-processing techniques in a single, adaptive pipeline:

1. **Moving-average trend decomposition** — separates trend from residual.
2. **Discrete Wavelet Transform (DWT) decomposition** — decomposes the residual into an approximation and multiple detail sub-bands.
3. **Fourier-domain processing** — amplifies dominant frequencies in the approximation and applies adaptive masking to detail coefficients.

The augmented residual is recomposed with the trend and then linearly mixed with the original sample using a Beta(0.5, 0.5) coefficient. This produces augmented training samples that preserve long-range structure while diversifying high-frequency variation.

This repository provides:
- The proposed **WaveFreqAug** method with full 5D hyperparameter grid search (`wave_freq/`).
- Reproductions of **six baseline augmentation methods** for fair comparison (`baseline/`): No Augmentation, Freq-Mask, Freq-Mix, Wave-Mask, Wave-Mix, STAug, and Dominant-Shuffle.
- An **ablation study** that isolates the contribution of each algorithmic component (`wave_freq/ablation_study/`).
- **Comparison visualizations** including MSE heatmaps, win-rate charts, styled line plots, pairwise scatter plots, and a Critical Difference diagram (`result_analysis/`).

---

## Algorithm: WaveFreqAug

```
Input:  x (batch_size, seq_len, enc_in)
        y (batch_size, pred_len, enc_in)

1.  Concatenate x and y along the time axis → xy (batch_size, total_len, enc_in)
2.  For each sample b and channel c:
    a. Compute moving-average trend with window W; residual = series - trend
    b. Apply pywt.wavedec(residual, wavelet, level=L) →
          approx (approximation coefficients)
          details[0..L-1] (detail coefficients per sub-band)
    c. Fourier enhancement on approx:
          FFT → amplify top top_k_ratio fraction of frequencies by 1.2× → IFFT
    d. Adaptive masking on each detail[i]:
          energy  = mean(|detail[i]|)
          adapted_mask_rate = mask_rate × (1 - energy / max(|detail[i]|))
          FFT → zero-out adapted_mask_rate fraction of frequencies → IFFT
    e. Reconstruct residual via pywt.waverec([approx] + details, wavelet)
    f. Recompose: xy_aug[b,:,c] = trend + reconstructed_residual
3.  λ ~ Beta(0.5, 0.5)  [or Uniform(0,1) for "uniform" mode]
4.  Return λ × xy_aug + (1 − λ) × xy
```

During training, the augmented loss is blended with the original loss:

```
loss_total = loss_original / 2 + loss_augmented / 2
```

Only a `sampling_rate` fraction of each batch is augmented per step, keeping compute overhead low.

---

## Repository Structure

```
WaveFreqAug_Forecasting/
├── dataset/                          # CSV datasets (not tracked by git)
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── national_illness.csv
│   └── weather.csv
│
├── wave_freq/                        # Proposed WaveFreqAug method
│   ├── main.py                       # Entry point: parallel 5D grid search
│   ├── utils/
│   │   ├── aug_method.py             # WaveFreqAug core implementation
│   │   ├── train_eval.py             # Training loop, validation, test (mixed precision)
│   │   ├── model.py                  # DLinear, SCINet, iTransformer
│   │   ├── dataloader.py             # TimeSeriesDataset
│   │   └── dataset_parameter.py     # Dataset configs (absolute paths — update these)
│   ├── ablation_study/
│   │   ├── aug_method_ablation.py   # Three augmentation variants for ablation
│   │   ├── ablation_main.py         # Ablation experiment runner
│   │   └── result/                  # Ablation CSVs per model
│   ├── checkpoints/                  # Saved model weights (per hyperparameter combo)
│   ├── results/                      # (legacy) per-run CSVs
│   └── plots/                        # Prediction plots
│
├── baseline/                         # Baseline augmentation methods
│   ├── base_main.py                  # Baseline experiment entry point
│   └── utils/
│       ├── aug_methods.py            # Freq-Mask, Freq-Mix, Wave-Mask, Wave-Mix, STAug, Dominant-Shuffle
│       ├── train_eval.py             # Training loop for baselines
│       ├── model.py                  # Same three forecast models
│       ├── dataloader.py             # Same dataset loader
│       └── dataset_parameter.py     # Dataset configs for baselines
│
├── results/
│   ├── baseline/{model}/            # Baseline experiment CSVs
│   └── wavefreq/{model}/            # WaveFreqAug experiment CSVs
│
├── result_analysis/
│   ├── plotting_mse_comparison.py   # All comparison figures (heatmap, bar, scatter, CD)
│   ├── result_analysis.ipynb        # Post-experiment metric aggregation notebook
│   ├── plotting_Signal_evaluation.ipynb  # Signal-level prediction visualization
│   └── analysis/                    # Generated figures (PNG/PDF) and win-count CSVs
│
├── wave_aug/                         # Python virtual environment (not tracked)
├── requirements.txt
├── CLAUDE.md
└── README.md
```

---

## Environment Setup

**Requirements:** Python 3.12, CUDA-capable GPU (tested with torch 2.1+cu126).

```bash
# Clone the repository
git clone <repo-url>
cd WaveFreqAug_Forecasting

# Create and activate a virtual environment
python3.12 -m venv wave_aug
source wave_aug/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA support (adjust the CUDA version to match your driver)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

> **Note:** CUDA is required at runtime. `wave_freq/utils/train_eval.py` hardcodes `GradScaler("cuda")` and `autocast(device_type="cuda")`.

---

## Datasets

Download the following datasets and place the CSV files in `dataset/`:

| Dataset | File | Channels | Frequency |
|---|---|---|---|
| ETTh1 | `ETTh1.csv` | 7 | Hourly |
| ETTh2 | `ETTh2.csv` | 7 | Hourly |
| ILI (Illness) | `national_illness.csv` | 7 | Weekly |
| Weather | `weather.csv` | 21 | 10-minute |

Standard train/val/test splits:

| Dataset | Train | Val | Test |
|---|---|---|---|
| ETTh1/ETTh2 | 12 months | 4 months | 4 months |
| ILI | 676 samples | 97 samples | 193 samples |
| Weather | 70% | 15% | 15% |

All splits are scaled using `StandardScaler` fitted on the training portion only.

---

## Path Configuration

`wave_freq/utils/dataset_parameter.py` and `baseline/utils/dataset_parameter.py` contain **hardcoded absolute paths**. Update the `data_path` entries before running on any machine:

```python
# wave_freq/utils/dataset_parameter.py
dataset_configs = {
    "ETTh1": {
        "data_path": "/your/path/to/dataset/ETTh1.csv",
        ...
    },
    ...
}
```

---

## Running WaveFreqAug Experiments

```bash
cd wave_freq
python main.py
```

`main.py` launches a `multiprocessing.Pool` with `NUM_WORKERS_POOL = 8` parallel workers (spawn context, `maxtasksperchild=1` for CUDA safety). Already-completed hyperparameter combinations are skipped automatically by checking for the checkpoint file.

### Hyperparameter Grid (5D, 486 combos per dataset/model/horizon)

| Parameter | Values |
|---|---|
| `mask_rate` | [0.1, 0.15, 0.2] |
| `level` (DWT depth) | [1, 3, 5] |
| `wavelet` | ["db2", "db4", "sym4"] |
| `lambd` (mixing shape) | ["U-Shape" → Beta(0.5,0.5), "uniform" → Uniform(0,1)] |
| `window` (trend MA window) | [6, 12, 24] |

`SAMPLING_RATE = 0.2` is fixed (batch-subsample fraction and `top_k_ratio`).

To limit VRAM usage with multiple workers sharing one GPU, set `BATCH_SIZE_OVERRIDE` at the top of `main.py` (e.g., `16` or `8`).

### Training Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss function | Smooth L1 (Huber) |
| Initial learning rate | 0.01 |
| LR schedule | Halved every epoch |
| Mixed precision | `torch.amp` (fp16 on CUDA) |
| Early stopping patience | 5 epochs |
| Batch size | 32 |
| Iterations per combo | 5 (reported as mean ± std) |

---

## Running Baseline Experiments

```bash
cd baseline
python base_main.py
```

Evaluates six SOTA augmentation strategies across all datasets and models:

| Aug Type | Description |
|---|---|
| `None` | No augmentation — pure supervised baseline |
| `Freq-Mask` | Random masking of Fourier frequencies (FrAug, Chen et al. 2023) |
| `Freq-Mix` | Cross-sample frequency mixing (FrAug) |
| `Wave-Mask` | Wavelet-domain coefficient masking (Wave-Augs) |
| `Wave-Mix` | Wavelet-domain coefficient mixing (Wave-Augs) |
| `StAug` | EMD seasonal-trend decomposition + Mixup (Zhang et al.) |
| `Dominant-Shuffle` | Dominant frequency shuffling |

---

## Ablation Study

The ablation study tests the contribution of each algorithmic component by selectively removing it, using the best hyperparameters found in the grid search for each `(dataset, model, pred_len)` combination.

```bash
cd wave_freq
python ablation_study/ablation_main.py
```

| Variant | Fourier Enhancement | Adaptive Masking |
|---|---|---|
| `WaveFreqAug-NoFourier` | ❌ removed | ✅ kept |
| `WaveFreqAug-NoMasking` | ✅ kept | ❌ removed |

Results are saved to `wave_freq/ablation_study/result/{model}/average_results_{dataset}.csv`. Compare against the full WaveFreqAug numbers in `results/wavefreq/`.

---

## Forecast Models

Three models are implemented identically in `wave_freq/utils/model.py` and `baseline/utils/model.py`:

### DLinear
Decomposition-Linear. Applies moving-average decomposition then fits separate linear layers to the trend and seasonal components per channel.
```
seq_len=336, kernel_size=25, individual=False
```

### SCINet
Sample Convolution and Interaction Network. Recursively downsamples into even/odd sub-sequences, applies cross-interaction convolutions, and concatenates for the forecast.
```
hid_size=1, num_stacks=1, num_levels=3, kernel_size=5, dropout=0.2
```

### iTransformer
Inverted Transformer. Applies self-attention across the channel dimension (each token is one variable's full sequence) rather than the time dimension.
```
d_model=512, n_heads=8, e_layers=4, d_ff=2048, dropout=0.1
```

---

## Output Structure

### WaveFreqAug results

```
results/wavefreq/{model}/
├── iteration_results_{dataset}.csv   # per-iteration metrics
└── average_results_{dataset}.csv     # mean ± std across 5 iterations
```

**CSV schema (17 columns):**
```
dataset, model, pred_len, aug_type, mask_rate, level, wavelet, lambd, window,
val_loss, mae, mse, rse, mae_std, mse_std, rse_std, exec_time
```

Checkpoints:
```
wave_freq/checkpoints/{dataset}_Wave-Freq_{pred_len}_{model}_{mask_rate}_{level}_{wavelet}_{lambd}_{window}/checkpoint.pth
```

### Baseline results

```
results/baseline/{model}/
├── iteration_results_{dataset}.csv
└── average_results_{dataset}.csv
```

Same 17-column schema; `mask_rate`, `level`, `wavelet`, `lambd`, `window` are set to `N/A`.

### Ablation results

```
wave_freq/ablation_study/result/{model}/
├── iteration_results_{dataset}.csv
└── average_results_{dataset}.csv
```

Same schema with `aug_variant` in place of `aug_type`.

---

## Analysis and Figures

```bash
# Update base_dir at the top of the script to point to the local results/ directory
cd result_analysis
python plotting_mse_comparison.py
```

Generates per-model figures in `result_analysis/analysis/`:

| Figure | Description |
|---|---|
| `mse_comparison_heatmap_{dataset}_{model}` | % MSE change vs no-augmentation baseline |
| `mse_pct_improvement_{model}` | Grouped bar chart of % MSE improvement |
| `win_rate_{model}` | Stacked win/tie/loss bar chart vs baseline |
| `fancy_lineplot_{model}` | Styled MSE vs prediction horizon per dataset |
| `scatter_vs_sota_{model}` | Pairwise scatter: WaveFreqAug vs each SOTA method |
| `cd_diagram_{model}` | Critical Difference diagram (Demšar 2006, Wilcoxon + Holm α=0.05) |

Win-count summaries are saved to `result_analysis/analysis/scatter_win_summary_{model}.csv`.

---

## Metrics

| Metric | Formula | Notes |
|---|---|---|
| MAE | mean(&#124;pred − true&#124;) | Mean Absolute Error |
| MSE | mean((pred − true)²) | Mean Squared Error |
| RSE | √(Σ(true−pred)²) / √(Σ(true−mean(true))²) | Relative Squared Error |

All metrics are computed on inverse-transformed (original scale) predictions.

---

## Integrating WaveFreqAug into Your Own Training Loop

```python
from wave_freq.utils.aug_method import Augmentation

aug = Augmentation()

# batch_x: (batch_size, seq_len, num_channels) — torch.Tensor (CPU)
# batch_y: (batch_size, pred_len, num_channels) — torch.Tensor (CPU)
xy_aug = aug.wave_freq_aug(
    batch_x,
    batch_y,
    mask_rate=0.15,    # fraction of detail frequencies to zero out
    wavelet="db4",     # pywt-compatible wavelet name
    level=3,           # DWT decomposition depth
    lambd="U-Shape",   # "U-Shape" → Beta(0.5,0.5), "uniform" → Uniform(0,1)
    window=12,         # moving-average window for trend decomposition
    top_k_ratio=0.2,   # fraction of approx frequencies to amplify
)
# xy_aug: (batch_size, seq_len + pred_len, num_channels)
x_aug = xy_aug[:, :seq_len, :]
y_aug = xy_aug[:, seq_len:, :]
```

---

## Baseline Method References

- **Freq-Mask / Freq-Mix** — Chen, M., Xu, Z., Zeng, A., & Xu, Q. (2023). *FrAug: Frequency Domain Augmentation for Time Series Forecasting*. arXiv:2302.09292.
- **Wave-Mask / Wave-Mix** — Bakhshaliyev et al. *Wave-Augs*. [github.com/jafarbakhshaliyev/Wave-Augs](https://github.com/jafarbakhshaliyev/Wave-Augs)
- **StAug** — Zhang, X. et al. *STAug*. [github.com/xiyuanzh/STAug](https://github.com/xiyuanzh/STAug)
- **DLinear** — Zeng, A. et al. (2023). *Are Transformers Effective for Time Series Forecasting?* AAAI 2023.
- **SCINet** — Liu, M. et al. (2022). *SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction Networks*. NeurIPS 2022.
- **iTransformer** — Liu, Y. et al. (2024). *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting*. ICLR 2024.
