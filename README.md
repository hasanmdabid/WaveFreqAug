# WaveFreqAug: Hybrid Wavelet-Frequency Data Augmentation for Time Series Forecasting

![WaveFreqAug Overview](figures/Flow_diagram.png)

**WaveFreqAug** is a novel data augmentation framework that combines **multi-resolution wavelet decomposition** with **adaptive frequency-domain masking and enhancement** to generate diverse yet temporally coherent synthetic time series. It significantly improves the generalization of deep forecasting models (e.g., DLinear, SCINet) across benchmark datasets such as **ETTh1, ETTh2, ILI, and Weather**.

> **Paper**: *WaveFreqAug: Hybrid Wavelet-Frequency Augmentation for Robust Time Series Forecasting*  
> **Journal**: [Under Review]  
> **License**: MIT

---

## Key Features

- **Preserves temporal coherence** by operating on the residual after trend removal.
- **Scale-specific augmentation** via Discrete Wavelet Transform (DWT).
- **Adaptive frequency masking** on detail coefficients based on energy.
- **Fourier enhancement** of low-frequency approximation for structural consistency.
- **Beta mixing** (`β(0.5, 0.5)`) for balanced interpolation.
- Outperforms **Base, STAug, Freq-Mask, Freq-Mix, Wave-Mask, Wave-Mix** in MSE across all prediction horizons.

---

## Method Overview

```text
Input Time Series X
    ↓
[Trend Decomposition] → trend + residual r
    ↓
[Wavelet Decomposition] → A_L, D_L, ..., D_1
    ↓
[Adaptive Masking] → D_l' (FFT → mask → IFFT)
[ Fourier Enhancement ] → A_L' (FFT → amplify top-k → IFFT)
    ↓
[Inverse DWT] → r'
    ↓
[Reconstruction] → x' = trend + r'
    ↓
[Beta Mixing] → x_aug = λ x' + (1−λ) X, λ ~ β(0.5, 0.5)
```

## Installation

Follow the the following instruction to successfully setup the python environment and dependencies. 

### Clone the repository

```text
git clone https://github.com/hasanmdabid/WaveFreqAug.git
cd WaveFreqAug
```

### Create a conda environment (recommended)

```text
conda create -n wavefreqaug python=3.9
conda activate wavefreqaug
```

### Install dependencies

```text
pip install -r requirements.txt
```

## Reproducibility

```text
Seeds: All experiments use seeds [42, 123, 456].
Hardware: Trained on NVIDIA RTX 3090 (24GB), 100 epochs, batch size 32.
Best configs per dataset/model are saved in analysis/best_wavefreq_*.csv.
Statistical tests: Friedman + Nemenyi (p < 0.05) confirm significance.
```

## Instruction to run the experiment

```text
To run and evaluate the models without any augmentation and contempoprary methods go to baselinefolder and run the base_main.py. To evaluate the WaveFreqAug method, go to wave_freq_aug folder and run wave_freq_aug.py script. 
```

## Citation

```text
@jpurnal{yourname2025wavefreqaug,
  title={WaveFreqAug: Hybrid Wavelet-Frequency Augmentation for Robust Time Series Forecasting},
  author={Hasan Et al.},
  booktitle={Under Review},
  year={2025}
}
```