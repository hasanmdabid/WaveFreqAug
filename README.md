# WaveFreqAug: Wavelet Guided Frequency Augmentation for Robust Time Series Forecasting

## Overview

**WaveFreqAug** is a data augmentation algorithm for multivariate time series forecasting. It combines three signal-processing techniques in a single, adaptive pipeline:

1. **Moving-average trend decomposition** — separates trend from residual.
2. **Discrete Wavelet Transform (DWT) decomposition** — decomposes the residual into an approximation and multiple detail sub-bands.
3. **Fourier-domain processing** — amplifies dominant frequencies in the approximation and applies adaptive masking to detail coefficients.

The augmented residual is recomposed with the trend and then linearly mixed with the original sample using a Beta(0.5, 0.5) coefficient. This produces augmented training samples that preserve long-range structure while diversifying high-frequency variation.

This repository provides:
- The proposed **WaveFreqAug** method (`wave_freq/`).
- Reproductions of five baseline augmentation methods for fair comparison (`baseline/`): No Augmentation, Freq-Mask, Freq-Mix, Wave-Mask, Wave-Mix, and STAug.

---

## Algorithm: WaveFreqAug

```
Input:  x (batch_size, seq_len, enc_in)
        y (batch_size, pred_len, enc_in)

1.  Concatenate x and y along the time axis → xy (batch_size, total_len, enc_in)
2.  For each sample b and channel c:
    a. Compute moving-average trend with window=12; residual = series - trend
    b. Apply pywt.wavedec(residual, wavelet, level=level) →
          approx (approximation coefficients)
          details[0..level-1] (detail coefficients per sub-band)
    c. Fourier enhancement on approx:
          FFT → amplify top top_k_ratio fraction of frequencies by 1.2× → IFFT
    d. Adaptive masking on each detail[i]:
          energy  = mean(|detail[i]|)
          adapted_mask_rate = mask_rate × (1 - energy / max(|detail[i]|))
          FFT → zero-out adapted_mask_rate fraction of frequencies → IFFT
    e. Reconstruct residual via pywt.waverec([approx] + details, wavelet)
    f. Recompose: xy_aug[b,:,c] = trend + reconstructed_residual
3.  λ ~ Beta(0.5, 0.5)
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
├── dataset/                        # CSV datasets (not tracked by git)
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── national_illness.csv
│   └── weather.csv
│
├── wave_freq/                      # Proposed WaveFreqAug method
│   ├── wave_freq_main.py           # Experiment entry point (grid search)
│   ├── aug_method.py               # WaveFreqAug core implementation
│   ├── train_eval.py               # Training loop, validation, test
│   ├── model.py                    # DLinear, SCINet, iTransformer
│   ├── dataloader.py               # TimeSeriesDataset
│   ├── dataset_parameter.py        # Dataset configs (absolute paths — update these)
│   ├── all_dataset_parameter.py    # Same configs with relative paths
│   ├── checkpoints/                # Saved model weights (per config)
│   ├── results/                    # CSV result files
│   └── plots/                      # Prediction plots
│
├── baseline/                       # Baseline augmentation methods
│   ├── base_main.py                # Baseline experiment entry point
│   ├── aug_methods.py              # Freq-Mask, Freq-Mix, Wave-Mask, Wave-Mix, STAug
│   ├── train_eval.py               # Training loop for baselines
│   ├── model.py                    # Same three forecast models
│   ├── dataloader.py               # Same dataset loader
│   ├── dataset_parameter.py        # Dataset configs for baselines
│   ├── checkpoints/                # Saved model weights per aug type
│   ├── results/                    # CSV result files
│   └── plots/                      # Prediction plots
│
├── wave_aug/                       # Python virtual environment (not tracked)
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

---

## Datasets

Download the following datasets and place the CSV files in `dataset/`:

| Dataset | File | Channels | Frequency |
|---|---|---|---|
| ETTh1 | `ETTh1.csv` | 7 | Hourly |
| ETTh2 | `ETTh2.csv` | 7 | Hourly |
| ILI (Illness) | `national_illness.csv` | 7 | Weekly |
| Weather | `weather.csv` | 21 | 10-minute |

Standard train/val/test splits used:

| Dataset | Train | Val | Test |
|---|---|---|---|
| ETTh1/ETTh2 | 12 months | 4 months | 4 months |
| ILI | 676 samples | 97 samples | 193 samples |
| Weather | 70% | 15% | 15% |

All splits are scaled using `StandardScaler` fitted on the training portion only.

---

## Path Configuration

Both `wave_freq/dataset_parameter.py` and `baseline/dataset_parameter.py` contain **hardcoded absolute paths**. Before running experiments on any machine, update the `data_path` entries to match your local dataset directory:

```python
# wave_freq/dataset_parameter.py  (and baseline/dataset_parameter.py)
dataset_configs = {
    "ETTh1": {
        "data_path": "/your/path/to/dataset/ETTh1.csv",
        ...
    },
    ...
}
```

Alternatively, `wave_freq/all_dataset_parameter.py` uses relative paths (`./dataset/`) and can be used by changing the import at the top of `wave_freq_main.py` and `wave_freq/train_eval.py`:

```python
# Change this line in wave_freq_main.py and wave_freq/train_eval.py
from all_dataset_parameter import dataset_configs   # relative-path version
```

---

## Running WaveFreqAug Experiments

All commands must be run from inside the `wave_freq/` directory because paths resolve relative to the script location.

```bash
cd wave_freq
python wave_freq_main.py
```

### What the experiment does

The experiment iterates over all combinations of:

- **Datasets**: ETTh1, ETTh2, ILI, Weather
- **Models**: iTransformer, DLinear, SCINet
- **Prediction lengths**:
  - ETTh1/ETTh2/Weather: {96, 192, 336, 720}
  - ILI: {24, 36, 48, 60}
- **Augmentation hyperparameter grid** (3 × 3 × 3 = 27 combinations per config):
  - `mask_rate`: [0.1, 0.15, 0.2]
  - `level` (wavelet decomposition depth): [1, 3, 5]
  - `wavelet`: ["db2", "db4", "sym4"]
- **Iterations per combination**: 3 (results reported as mean ± std)

### Selecting a subset for a quick run

Edit the `main()` call at the bottom of `wave_freq_main.py`:

```python
# Example: single model, 5 epochs, 1 iteration
models = ["DLinear"]
main(models, epochs=5, learning_rate=0.01, patience=3, num_iterations=1, label_len=0)
```

To restrict to a single dataset, edit `dataset_parameter.py` to include only that dataset's entry in `dataset_configs`.

### Training hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss function | Smooth L1 (Huber) |
| Initial learning rate | 0.01 |
| LR schedule | Halved every epoch |
| Mixed precision | `torch.amp` (fp16 on CUDA) |
| Early stopping patience | 5 epochs |
| Batch size | 32 |
| Sampling rate | 0.2 (20% of each batch is augmented) |
| `top_k_ratio` | 0.2 (top 20% frequencies amplified in approx) |
| Moving-average window | 12 |

---

## Running Baseline Experiments

```bash
cd baseline
python base_main.py
```

The baseline evaluates six augmentation strategies across the same datasets and models:

| Aug Type | Description |
|---|---|
| `None` | No augmentation — pure supervised baseline |
| `Freq-Mask` | Random masking of Fourier frequencies (FrAug, Chen et al. 2023) |
| `Freq-Mix` | Cross-sample frequency mixing, preserving dominant components (FrAug) |
| `Wave-Mask` | Wavelet-domain coefficient masking (Wave-Augs, Bakhshaliyev et al.) |
| `Wave-Mix` | Wavelet-domain coefficient mixing across samples (Wave-Augs) |
| `STAug` | EMD seasonal-trend decomposition + Mixup (STAug, Zhang et al.) |

Augmentation hyperparameters per dataset × prediction length are defined in `baseline/dataset_parameter.py` under the `"aug_params"` key.

---

## Forecast Models

Three models are implemented in `model.py` (identical in both `wave_freq/` and `baseline/`):

### DLinear
Decomposition-Linear model. Applies a moving-average decomposition then fits separate linear layers to the trend and seasonal components per channel. Fast and strong linear baseline.

```
seq_len=336, kernel_size=25, individual=False
```

### SCINet
Sample Convolution and Interaction Network. Recursively downsamples into even/odd sub-sequences, processes them with cross-interaction convolutions, and concatenates for the forecast.

```
hid_size=1, num_stacks=1, num_levels=3, kernel_size=5, dropout=0.2
```


```
d_model=512, n_heads=8, e_layers=4, d_ff=2048, dropout=0.1
```

---

## Output Structure

### WaveFreqAug (`wave_freq/`)

```
wave_freq/
├── checkpoints/{dataset}_{aug_type}_{pred_len}_{model}_{mask_rate}_{level}_{wavelet}/
│   └── checkpoint.pth              # Best model weights for this config
├── results/
│   ├── iteration_results_{dataset}.csv   # MAE, MSE, RSE for every iteration
│   └── average_results_{dataset}.csv     # Mean ± std across iterations
└── plots/
    └── prediction_{dataset}_Wave-Freq_{pred_len}_{model}_{mask_rate}_{level}_{wavelet}.png
```

**`iteration_results_{dataset}.csv` columns:**
```
dataset, model, pred_len, aug_type, mask_rate, level, wavelet, iteration, val_loss, mae, mse, rse
```

**`average_results_{dataset}.csv` columns:**
```
dataset, model, pred_len, aug_type, mask_rate, level, wavelet, val_loss, mae, mse, rse, mae_std, mse_std, rse_std
```

### Baseline (`baseline/`)

```
baseline/
├── checkpoints/{aug_type}/checkpoint.pth
├── results/
│   ├── iteration_results_{dataset}.csv
│   └── average_results_{dataset}.csv
└── plots/Dlinear/
    └── prediction_{dataset}_{aug_type}_{pred_len}.png
```

**`average_results_{dataset}.csv` columns** (baseline, no hyperparameter columns):
```
dataset, model, pred_len, aug_type, val_loss, mae, mse, rse, mae_std, mse_std, rse_std
```

---

## Metrics

| Metric | Formula | Notes |
|---|---|---|
| MAE | mean(&#124;pred − true&#124;) | Mean Absolute Error |
| MSE | mean((pred − true)²) | Mean Squared Error |
| RSE | √(Σ(true−pred)²) / √(Σ(true−mean(true))²) | Relative Squared Error |

All metrics are computed on inverse-transformed (original scale) predictions.

---

## Extending to New Datasets

1. Add the CSV to `dataset/`. The first column must be a timestamp or index; subsequent columns are feature channels.
2. Add an entry to `wave_freq/dataset_parameter.py`:

```python
"MyDataset": {
    "data_path": "/absolute/path/to/dataset/my_dataset.csv",
    "data_name": "MyDataset",
    "seq_len": 336,
    "pred_lens": [96, 192, 336, 720],
    "enc_in": <number_of_channels>,
    "batch_size": 32,
    "aug_types": ["Wave-Freq"],
    "aug_params": { ... },  # only used by baseline; wave_freq uses grid search
},
```

3. Extend `dataloader.py` to handle the split logic for your dataset name inside `__read_data__`.

---

## Integrating WaveFreqAug into Your Own Training Loop

The augmentation class is self-contained in `wave_freq/aug_method.py`. To use it independently:

```python
from aug_method import Augmentation

aug = Augmentation()

# batch_x: (batch_size, seq_len, num_channels) — torch.Tensor
# batch_y: (batch_size, pred_len, num_channels) — torch.Tensor
xy_aug = aug.wave_freq_aug(
    batch_x,
    batch_y,
    mask_rate=0.15,   # fraction of detail frequencies to zero out
    wavelet="db4",    # pywt-compatible wavelet name
    level=3,          # DWT decomposition depth
    lambd=None,       # None → sample from Beta(0.5, 0.5)
    window=12,        # moving-average window for trend decomposition
    top_k_ratio=0.2,  # fraction of approx frequencies to amplify
)
# xy_aug: (batch_size, seq_len + pred_len, num_channels)
x_aug = xy_aug[:, :seq_len, :]
y_aug = xy_aug[:, seq_len:, :]
```

---

## Baseline Method References

- **Freq-Mask / Freq-Mix** — Chen, M., Xu, Z., Zeng, A., & Xu, Q. (2023). *FrAug: Frequency Domain Augmentation for Time Series Forecasting*. arXiv:2302.09292.
- **Wave-Mask / Wave-Mix** — Bakhshaliyev et al. *Wave-Augs*. [github.com/jafarbakhshaliyev/Wave-Augs](https://github.com/jafarbakhshaliyev/Wave-Augs)
- **STAug** — Zhang, X. et al. *STAug*. [github.com/xiyuanzh/STAug](https://github.com/xiyuanzh/STAug)
- **DLinear** — Zeng, A. et al. (2023). *Are Transformers Effective for Time Series Forecasting?* AAAI 2023.
- **SCINet** — Liu, M. et al. (2022). *SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction Networks*. NeurIPS 2022.

