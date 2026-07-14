import torch
import numpy as np
import pywt
from numpy.fft import fft, ifft


def _sample_lambd(lambd):
    if lambd == "uniform":
        return np.random.uniform(0, 1)
    return np.random.beta(0.5, 0.5)  # "U-Shape"


class AblationAugmentation:
    """Three variants of WaveFreqAug for ablation study.

    Full       — complete algorithm (reference)
    NoFourier  — Fourier enhancement on approximation removed
    NoMasking  — adaptive masking on detail coefficients removed
    """

    def wave_freq_aug_full(
        self,
        x,
        y,
        mask_rate=0.15,
        wavelet="db4",
        level=4,
        lambd="U-Shape",
        window=12,
        top_k_ratio=0.2,
    ):
        """Complete WaveFreqAug: trend decomp + Fourier enh + adaptive masking."""
        xy = torch.cat([x, y], dim=1)
        batch_size, total_len, enc_in = xy.shape
        xy_np = xy.cpu().numpy()
        xy_aug = np.zeros_like(xy_np)
        lam = _sample_lambd(lambd)

        for b in range(batch_size):
            for c in range(enc_in):
                series = xy_np[b, :, c]
                trend = np.convolve(series, np.ones(window) / window, mode="same")
                residual = series - trend
                coeffs = pywt.wavedec(residual, wavelet, level=level, axis=0)
                approx = coeffs[0]
                details = list(coeffs[1:])

                # Fourier enhancement on approximation
                freq_approx = fft(approx)
                top_k = max(1, int(len(freq_approx) * top_k_ratio))
                top_idx = np.argsort(np.abs(freq_approx))[-top_k:]
                freq_approx[top_idx] *= 1.2
                approx = np.real(ifft(freq_approx))

                # Adaptive masking on detail coefficients
                for i in range(len(details)):
                    energy = np.mean(np.abs(details[i]))
                    dmax = np.max(np.abs(details[i]))
                    adapted = mask_rate * (1 - energy / dmax) if dmax > 0 else 0.0
                    freq = fft(details[i])
                    freq[np.random.rand(len(freq)) < adapted] = 0
                    details[i] = np.real(ifft(freq))

                max_level = min(level, pywt.dwt_max_level(len(residual), wavelet))
                if max_level < level:
                    all_coeffs = (
                        [approx] + details[:max_level] + [None] * (level - max_level)
                    )
                else:
                    all_coeffs = [approx] + details
                residual_aug = pywt.waverec(all_coeffs, wavelet, axis=0)[:total_len]
                xy_aug[b, :, c] = trend + residual_aug

        return torch.tensor(lam * xy_aug + (1 - lam) * xy_np, dtype=torch.float32)

    def wave_freq_aug_no_fourier(
        self,
        x,
        y,
        mask_rate=0.15,
        wavelet="db4",
        level=4,
        lambd="U-Shape",
        window=12,
        top_k_ratio=0.2,
    ):
        """WaveFreqAug without Fourier enhancement on the approximation coefficient."""
        xy = torch.cat([x, y], dim=1)
        batch_size, total_len, enc_in = xy.shape
        xy_np = xy.cpu().numpy()
        xy_aug = np.zeros_like(xy_np)
        lam = _sample_lambd(lambd)

        for b in range(batch_size):
            for c in range(enc_in):
                series = xy_np[b, :, c]
                trend = np.convolve(series, np.ones(window) / window, mode="same")
                residual = series - trend
                coeffs = pywt.wavedec(residual, wavelet, level=level, axis=0)
                approx = coeffs[0]  # no Fourier enhancement — passed through as-is
                details = list(coeffs[1:])

                # Adaptive masking on detail coefficients (kept)
                for i in range(len(details)):
                    energy = np.mean(np.abs(details[i]))
                    dmax = np.max(np.abs(details[i]))
                    adapted = mask_rate * (1 - energy / dmax) if dmax > 0 else 0.0
                    freq = fft(details[i])
                    freq[np.random.rand(len(freq)) < adapted] = 0
                    details[i] = np.real(ifft(freq))

                max_level = min(level, pywt.dwt_max_level(len(residual), wavelet))
                if max_level < level:
                    all_coeffs = (
                        [approx] + details[:max_level] + [None] * (level - max_level)
                    )
                else:
                    all_coeffs = [approx] + details
                residual_aug = pywt.waverec(all_coeffs, wavelet, axis=0)[:total_len]
                xy_aug[b, :, c] = trend + residual_aug

        return torch.tensor(lam * xy_aug + (1 - lam) * xy_np, dtype=torch.float32)

    def wave_freq_aug_no_masking(
        self,
        x,
        y,
        mask_rate=0.15,
        wavelet="db4",
        level=4,
        lambd="U-Shape",
        window=12,
        top_k_ratio=0.2,
    ):
        """WaveFreqAug without adaptive masking on the detail coefficients."""
        xy = torch.cat([x, y], dim=1)
        batch_size, total_len, enc_in = xy.shape
        xy_np = xy.cpu().numpy()
        xy_aug = np.zeros_like(xy_np)
        lam = _sample_lambd(lambd)

        for b in range(batch_size):
            for c in range(enc_in):
                series = xy_np[b, :, c]
                trend = np.convolve(series, np.ones(window) / window, mode="same")
                residual = series - trend
                coeffs = pywt.wavedec(residual, wavelet, level=level, axis=0)
                approx = coeffs[0]
                details = list(coeffs[1:])  # no masking — passed through as-is

                # Fourier enhancement on approximation (kept)
                freq_approx = fft(approx)
                top_k = max(1, int(len(freq_approx) * top_k_ratio))
                top_idx = np.argsort(np.abs(freq_approx))[-top_k:]
                freq_approx[top_idx] *= 1.2
                approx = np.real(ifft(freq_approx))

                max_level = min(level, pywt.dwt_max_level(len(residual), wavelet))
                if max_level < level:
                    all_coeffs = (
                        [approx] + details[:max_level] + [None] * (level - max_level)
                    )
                else:
                    all_coeffs = [approx] + details
                residual_aug = pywt.waverec(all_coeffs, wavelet, axis=0)[:total_len]
                xy_aug[b, :, c] = trend + residual_aug

        return torch.tensor(lam * xy_aug + (1 - lam) * xy_np, dtype=torch.float32)
