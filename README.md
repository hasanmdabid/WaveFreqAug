# WaveFreqAug: Wavelet Guided Frequency Augmentation for Robust Time Series Forecasting

## Overview

**WaveFreqAug** is a data augmentation algorithm for multivariate time series forecasting. It combines three signal-processing techniques in a single, adaptive pipeline:

1. **Moving-average trend decomposition** — separates trend from residual.
2. **Discrete Wavelet Transform (DWT) decomposition** — decomposes the residual into an approximation and multiple detail sub-bands.
3. **Fourier-domain processing** — amplifies dominant frequencies in the approximation and applies adaptive masking to detail coefficients.

The augmented residual is recomposed with the trend and then linearly mixed with the original sample using a Beta(0.5, 0.5) coefficient. This produces augmented training samples that preserve long-range structure while diversifying high-frequency variation.

This repository provides:
- The proposed **WaveFreqAug** method, evaluated across **six forecasting models** via hyperparameter search (`main/`).
- Reproductions of **six baseline augmentation methods** for fair comparison (`baseline/`): No Augmentation, Freq-Mask, Freq-Mix, Wave-Mask, Wave-Mix, StAug, and Dominant-Shuffle.

---

## Algorithm: WaveFreqAug

```
Require:  Time series X = [x; y] ∈ R^(T × C) (concatenation of the look-back
          window x and the forecast horizon y), window size w, decomposition
          level L, wavelet type W, base masking rate m_r, ratio top_k_ratio,
          mixing strategy λ_mode ∈ {U-Shape, uniform}
Ensure:   Augmented series x_aug ∈ R^(T × C) (the augmented input and
          augmented label are recovered by splitting x_aug at T_in)

Step 1 — Trend decomposition (causal moving average):
    τ_t ← (1/w) · Σ_{i=t-w+1}^{t} X_i,   t = 1, ..., T
        # backward-looking only — no future (label-side) value ever enters τ
    r ← X − τ

Step 2 — Wavelet decomposition:
    [A_L, D_L, D_{L-1}, ..., D_1] ← DWT(r, L, W)

Step 3 — Adaptive masking on detail coefficients, for l = 1 to L:
    M_l ← mean(|D_l|)                                    # mean magnitude
    P_l ← max(|D_l|)                                      # peak magnitude
    m_l ← m_r · (1 − M_l / P_l)  if P_l > 0,  else 0       # adaptive rate
    F_d ← FFT(D_l)
    F_d'[f] ← 0 with probability m_l, else F_d[f]
    D_l' ← Re{ IFFT(F_d') }

Step 4 — Fourier enhancement of the approximation:
    F_a ← FFT(A_L)
    k ← |F_a| · top_k_ratio
    F_a'[top_k_indices] ← F_a[top_k_indices] × α
    A_L' ← Re{ IFFT(F_a') }

Step 5 — Reconstruction and mixing:
    r' ← IDWT([A_L', D_L', D_{L-1}', ..., D_1'], W)
    x' ← τ + r'
    λ ~ Beta(0.5, 0.5)  if λ_mode = U-Shape,  else  λ ~ Uniform(0, 1)
    x_aug ← λ · x' + (1 − λ) · X

Return x_aug
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
├── dataset/                           # CSV datasets (not tracked by git)
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── national_illness.csv
│   ├── weather.csv
│   └── exchange_rate.csv
│
├── main/                               # Proposed WaveFreqAug method
│   ├── main.py                         # Entry point: parallel random/grid hyperparameter search
│   ├── checkpoints/                    # Saved model weights (per hyperparameter combo)
│   ├── plots/                          # Prediction plots
│   ├── results/
│   │   ├── grid_search/{model}/        # Full 5D grid-search results (DLinear/SCINet, legacy run)
│   │   └── random_search/{model}/      # Random-search results (all six models)
│   └── utils/
│       ├── aug_method.py               # WaveFreqAug core implementation
│       ├── train_eval.py               # Training loop, validation, test (mixed precision)
│       ├── model.py                    # DLinear, SCINet, iTransformer, FEDformer, TiDE, PatchTST
│       ├── dataloader.py               # TimeSeriesDataset
│       ├── dataset_parameter.py        # Dataset configs (absolute paths — update these)
│       └── helper.py                   # Shared per-combo train/eval loop + resume logic
│
├── baseline/                           # Baseline augmentation methods
│   ├── base_main.py                    # Baseline experiment entry point
│   ├── checkpoints/                    # Saved model weights (per combo)
│   ├── plots/                          # Prediction plots
│   ├── results/{model}/                # Baseline experiment CSVs (six models)
│   └── utils/
│       ├── aug_methods.py              # Freq-Mask, Freq-Mix, Wave-Mask, Wave-Mix, StAug, Dominant-Shuffle
│       ├── train_eval.py               # Training loop for baselines
│       ├── model.py                    # Same six forecast models
│       ├── dataloader.py               # Same dataset loader
│       ├── dataset_parameter.py        # Dataset configs for baselines
│       └── helper.py                   # Shared per-combo train/eval loop + resume logic
│
├── wave_aug/                           # Python virtual environment (not tracked)
├── requirements.txt
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

> **Note:** CUDA is required at runtime. `main/utils/train_eval.py` hardcodes `GradScaler("cuda")` and `autocast(device_type="cuda")`.

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

`main/utils/dataset_parameter.py` and `baseline/utils/dataset_parameter.py` contain **hardcoded absolute paths**. Update the `data_path` entries before running on any machine:

```python
# main/utils/dataset_parameter.py
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
cd main
python main.py
```

`main.py` launches a `multiprocessing.Pool` of parallel workers (spawn context, `maxtasksperchild=1` for CUDA safety). Already-completed hyperparameter combinations are skipped automatically by checking for a row in the model's `average_results_{dataset}.csv`.

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
| Iterations per combo | 3 (reported as mean ± std) |

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

## Forecast Models

Six models are implemented identically in `main/utils/model.py` and `baseline/utils/model.py`:

- **DLinear** — Zeng, A. et al. (2023). *Are Transformers Effective for Time Series Forecasting?* AAAI 2023.
- **SCINet** — Liu, M. et al. (2022). *SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction Networks*. NeurIPS 2022.
- **iTransformer** — Liu, Y. et al. (2024). *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting*. ICLR 2024.
- **FEDformer** — Zhou, T. et al. (2022). *FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting*. ICML 2022.
- **TiDE** — Das, A. et al. (2023). *Long-term Forecasting with TiDE: Time-series Dense Encoder*. arXiv:2304.08424.
- **PatchTST** — Nie, Y. et al. (2023). *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*. ICLR 2023.

---

## Output Structure

### WaveFreqAug results

```
main/results/{grid_search|random_search}/{model}/
├── iteration_results_{dataset}.csv   # per-iteration metrics
└── average_results_{dataset}.csv     # mean ± std across iterations
```

**CSV schema (17 columns):**
```
dataset, model, pred_len, aug_type, mask_rate, level, wavelet, lambd, window,
val_loss, mae, mse, rse, mae_std, mse_std, rse_std, exec_time
```

Checkpoints:
```
main/checkpoints/{grid_search|random_search}/{dataset}_Wave-Freq_{pred_len}_{model}_{mask_rate}_{level}_{wavelet}_{lambd}_{window}/checkpoint.pth
```

### Baseline results

```
baseline/results/{model}/
├── iteration_results_{dataset}.csv
└── average_results_{dataset}.csv
```

Same 17-column schema; `mask_rate`, `level`, `wavelet`, `lambd`, `window` are set to `N/A`.

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
from main.utils.aug_method import Augmentation

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
- **FEDformer** — Zhou, T. et al. (2022). *FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting*. ICML 2022.
- **TiDE** — Das, A. et al. (2023). *Long-term Forecasting with TiDE: Time-series Dense Encoder*. arXiv:2304.08424.
- **PatchTST** — Nie, Y. et al. (2023). *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*. ICLR 2023.
