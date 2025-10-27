import torch
import numpy as np
import pywt
from numpy.fft import fft, ifft


class Augmentation:
    def __init__(self):
        pass

    def wave_freq_aug(
        self,
        x,
        y,
        mask_rate=0.15,
        wavelet="db4",
        level=4,
        lambd=None,
        dim=1,
        window=12,
        top_k_ratio=0.2,
    ):
        """
        Improved WaveFreqAug with trend decomposition, adaptive masking, and Fourier enhancement.
        Args:
            x: Input tensor (batch_size, seq_len, enc_in)
            y: Ground truth (batch_size, pred_len, enc_in)
            mask_rate: Base masking rate (tuned lower)
            wavelet: Wavelet type (use smoother 'db4')
            level: Decomposition level (higher for finer details)
            lambd: Mixing coefficient (if None, Beta(0.5, 0.5))
            dim: Dimension for augmentation
            window: Moving average window for trend decomposition
            top_k_ratio: Ratio of top frequencies to amplify in approximation
        Returns:
            Augmented tensor (batch_size, seq_len + pred_len, enc_in)
        """
        xy = torch.cat([x, y], dim=1)  # (batch_size, total_len, enc_in)
        batch_size, total_len, enc_in = xy.shape
        xy_np = xy.cpu().numpy()
        xy_aug = np.zeros_like(xy_np)

        if lambd is None:
            lambd = np.random.beta(0.5, 0.5)

        for b in range(batch_size):
            for c in range(enc_in):
                series = xy_np[b, :, c]
                # Trend decomposition
                trend = np.convolve(series, np.ones(window) / window, mode="same")
                residual = series - trend

                # Wavelet decomposition on residual
                coeffs = pywt.wavedec(residual, wavelet, level=level, axis=0)
                approx = coeffs[0]
                details = coeffs[1:]

                # Fourier enhancement on approximation
                freq_approx = fft(approx)
                amp = np.abs(freq_approx)
                top_k = int(len(freq_approx) * top_k_ratio)
                top_indices = np.argsort(amp)[-top_k:]
                freq_approx[top_indices] *= 1.2  # Amplify top frequencies
                approx = np.real(ifft(freq_approx))

                # Adaptive masking on details
                for i in range(len(details)):
                    energy = np.mean(np.abs(details[i]))
                    adapted_mask_rate = mask_rate * (
                        1 - energy / np.max(np.abs(details[i]))
                    )  # Lower for high-energy
                    freq = fft(details[i])
                    mask = np.random.rand(len(freq)) < adapted_mask_rate
                    freq[mask] = 0
                    details[i] = np.real(ifft(freq))

                # Reconstruct residual
                max_level = min(level, pywt.dwt_max_level(len(residual), wavelet))
                if max_level < level:
                    coeffs = coeffs[: max_level + 1] + [None] * (level - max_level)
                residual_aug = pywt.waverec([approx] + details, wavelet, axis=0)[
                    :total_len
                ]

                # Recompose with trend
                xy_aug[b, :, c] = trend + residual_aug

        # Mix with original
        xy_aug = lambd * xy_aug + (1 - lambd) * xy_np
        return torch.tensor(xy_aug, dtype=torch.float32)
