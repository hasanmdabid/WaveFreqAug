# =============================================================================
# The freq_mask, freq_mix methods are adapted from the following sources:
#  Chen, M., Xu, Z., Zeng, A., & Xu, Q. (2023). "FrAug: Frequency Domain Augmentation for Time Series Forecasting".
#  arXiv preprint arXiv:2302.09292.
#
# The emd_aug and mix_aug for STAug method are adapted from the following source:
#  https://github.com/xiyuanzh/STAug/tree/main

# The wave_mask and wave_mix methods are adopted from the following source:
# https://github.com/jafarbakhshaliyev/Wave-Augs
# =============================================================================


import torch
import numpy as np
from pytorch_wavelets import DWT1DForward, DWT1DInverse
import pywt
from numpy.fft import fft, ifft

# Augmentation class
class Augmentation:
    @staticmethod
    def freq_mask(
        x: torch.Tensor, y: torch.Tensor, rate: float = 0.5, dim: int = 1
    ) -> torch.Tensor:
        xy = torch.cat([x, y], dim=dim)
        xy_f = torch.fft.rfft(xy, dim=dim)
        m = torch.rand(xy_f.shape, device=xy_f.device) < rate
        freal = xy_f.real.masked_fill(m, 0)
        fimag = xy_f.imag.masked_fill(m, 0)
        xy_f = torch.complex(freal, fimag)
        xy = torch.fft.irfft(xy_f, dim=dim)
        return xy

    @staticmethod
    def freq_mix(
        x: torch.Tensor, y: torch.Tensor, rate: float = 0.5, dim: int = 1
    ) -> torch.Tensor:
        xy = torch.cat([x, y], dim=dim)
        xy_f = torch.fft.rfft(xy, dim=dim)
        m = torch.rand(xy_f.shape, device=xy_f.device) < rate
        amp = abs(xy_f)
        _, index = amp.sort(dim=dim, descending=True)
        dominant_mask = index > 2
        m = torch.bitwise_and(m, dominant_mask)
        freal = xy_f.real.masked_fill(m, 0)
        fimag = xy_f.imag.masked_fill(m, 0)
        b_idx = np.arange(x.shape[0])
        np.random.shuffle(b_idx)
        x2, y2 = x[b_idx], y[b_idx]
        xy2 = torch.cat([x2, y2], dim=dim)
        xy2_f = torch.fft.rfft(xy2, dim=dim)
        m = torch.bitwise_not(m)
        freal2 = xy2_f.real.masked_fill(m, 0)
        fimag2 = xy2_f.imag.masked_fill(m, 0)
        freal += freal2
        fimag += fimag2
        xy_f = torch.complex(freal, fimag)
        xy = torch.fft.irfft(xy_f, dim=dim)
        return xy

    @staticmethod
    def wave_mask(
        x: torch.Tensor,
        y: torch.Tensor,
        rates: list,
        wavelet: str = "db1",
        level: int = 2,
        dim: int = 1,
    ) -> torch.Tensor:
        xy = torch.cat([x, y], dim=1)
        rate_tensor = torch.tensor(
            rates[: level + 1], device=x.device
        )  # Ensure rates match level
        xy = xy.permute(0, 2, 1).to(x.device).to(torch.float64)
        try:
            dwt = (
                DWT1DForward(J=level, wave=wavelet, mode="symmetric")
                .to(x.device)
                .double()
            )
        except:
            print(f"Wavelet {wavelet} not supported, falling back to db1")
            dwt = (
                DWT1DForward(J=level, wave="db1", mode="symmetric")
                .to(x.device)
                .double()
            )
        cA, cDs = dwt(xy)
        mask = torch.rand_like(cA).to(cA.device) < rate_tensor[0]
        cA = cA.masked_fill(mask, 0)
        masked_cDs = []
        for i, cD in enumerate(cDs, 1):
            mask_cD = torch.rand_like(cD).to(cD.device) < rate_tensor[i]
            cD = cD.masked_fill(mask_cD, 0)
            masked_cDs.append(cD)
        try:
            idwt = DWT1DInverse(wave=wavelet, mode="symmetric").to(x.device).double()
        except:
            idwt = DWT1DInverse(wave="db1", mode="symmetric").to(x.device).double()
        reconstructed = idwt((cA, masked_cDs))
        reconstructed = reconstructed.permute(0, 2, 1)
        return reconstructed.float()

    @staticmethod
    def wave_mix(
        x: torch.Tensor,
        y: torch.Tensor,
        rates: list,
        wavelet: str = "db1",
        level: int = 2,
        dim: int = 1,
    ) -> torch.Tensor:
        xy = torch.cat([x, y], dim=1)
        batch_size, _, _ = xy.shape
        rate_tensor = torch.tensor(rates[: level + 1], device=x.device)
        xy = xy.permute(0, 2, 1).to(x.device).to(torch.float64)
        b_idx = torch.randperm(batch_size)
        xy2 = xy[b_idx].to(torch.float64)
        try:
            dwt = (
                DWT1DForward(J=level, wave=wavelet, mode="symmetric")
                .to(x.device)
                .double()
            )
        except:
            print(f"Wavelet {wavelet} not supported, falling back to db1")
            dwt = (
                DWT1DForward(J=level, wave="db1", mode="symmetric")
                .to(x.device)
                .double()
            )
        cA1, cDs1 = dwt(xy)
        cA2, cDs2 = dwt(xy2)
        mask = torch.rand_like(cA1).to(cA1.device) < rate_tensor[0]
        cA_mixed = cA1.masked_fill(mask, 0) + cA2.masked_fill(~mask, 0)
        mixed_cDs = []
        for i, (cD1, cD2) in enumerate(zip(cDs1, cDs2), 1):
            mask = torch.rand_like(cD1).to(cD1.device) < rate_tensor[i]
            cD_mixed = cD1.masked_fill(mask, 0) + cD2.masked_fill(~mask, 0)
            mixed_cDs.append(cD_mixed)
        try:
            idwt = DWT1DInverse(wave=wavelet, mode="symmetric").to(x.device).double()
        except:
            idwt = DWT1DInverse(wave="db1", mode="symmetric").to(x.device).double()
        reconstructed = idwt((cA_mixed, mixed_cDs))
        reconstructed = reconstructed.permute(0, 2, 1)
        return reconstructed.float()

    @staticmethod
    def dominant_shuffle(
        x: torch.Tensor, y: torch.Tensor, k: int = 4, dim: int = 1
    ) -> torch.Tensor:
        """
        Dominant Shuffle augmentation (Zhao et al., 2024, arXiv:2405.16456).

        Concatenate [x, y], take the rFFT along the time axis, select the top-k
        dominant frequencies by amplitude (excluding the DC component), randomly
        permute those k complex components per (sample, channel), then iFFT back.
        Args:
            x: Input tensor (batch_size, seq_len, enc_in)
            y: Ground truth (batch_size, pred_len, enc_in)
            k: Number of dominant frequencies to shuffle
            dim: Time dimension (typically 1)
        Returns:
            Augmented tensor (batch_size, seq_len + pred_len, enc_in)
        """
        xy = torch.cat([x, y], dim=dim)
        total_len = xy.size(dim)
        xy_f = torch.fft.rfft(xy, dim=dim)
        magnitude = xy_f.abs()
        # Exclude DC (index 0); cap k to the number of available non-DC bins.
        n_freq = xy_f.size(dim)
        k = max(0, min(int(k), n_freq - 1))
        if k == 0:
            return xy.float()
        # (batch, k, enc_in): the k highest-amplitude frequency positions per sample/channel.
        topk_indices = torch.argsort(magnitude, dim=dim, descending=True)[:, 1 : k + 1, :]
        new_xy_f = xy_f.clone()
        batch_size, _, enc_in = xy.shape
        for c in range(enc_in):
            for b in range(batch_size):
                idx = topk_indices[b, :, c]            # original dominant positions
                perm = idx[torch.randperm(k)]          # same positions, shuffled order
                new_xy_f[b, idx, c] = xy_f[b, perm, c]  # permute the complex components
        xy = torch.fft.irfft(new_xy_f, n=total_len, dim=dim)
        return xy.float()

    @staticmethod
    def emd_aug(x: torch.Tensor) -> torch.Tensor:
        b, n_imf, t, c = x.size()
        inp = x.permute(0, 2, 1, 3).reshape(b, t, n_imf * c)
        if torch.rand(1) >= 0.5:
            w = 2 * torch.rand((b, 1, n_imf * c)).to(x.device)
        else:
            w = torch.ones((b, 1, n_imf * c)).to(x.device)
        w_exp = w.expand(-1, t, -1)
        out = w_exp * inp
        out = out.reshape(b, t, n_imf, c).sum(dim=2)
        return out

    @staticmethod
    def mix_aug(
        batch_x: torch.Tensor, batch_y: torch.Tensor, lambd: float = 0.5
    ) -> tuple:
        inds2 = torch.randperm(batch_x.shape[0])
        lam = np.random.beta(lambd, lambd)
        batch_x = lam * batch_x[inds2] + (1 - lam) * batch_x
        batch_y = lam * batch_y[inds2] + (1 - lam) * batch_y
        return batch_x, batch_y

    def wave_freq_aug(
        self, x, y, mask_rate=0.3, wavelet="db2", level=3, lambd=None, dim=1
    ):
        """
        Apply Wavelet-Frequency Augmentation (WaveFreqAug).
        Decomposes time series using DWT, applies frequency masking to detail coefficients,
        reconstructs with IDWT, and mixes with original series.
        Args:
            x: Input tensor (batch_size, seq_len, enc_in)
            y: Ground truth (batch_size, pred_len, enc_in)
            mask_rate: Fraction of frequencies to mask in detail coefficients
            wavelet: Wavelet type (e.g., 'db2')
            level: Decomposition level (default 3)
            lambd: Mixing coefficient (if None, sampled from Beta(0.5, 0.5))
            dim: Dimension to apply augmentation (typically 1 for time axis)
        Returns:
            Augmented tensor (batch_size, seq_len + pred_len, enc_in)
        """
        # Combine input and ground truth
        xy = torch.cat([x, y], dim=1)  # Shape: (batch_size, seq_len + pred_len, enc_in)
        batch_size, total_len, enc_in = xy.shape
        xy_np = xy.cpu().numpy()  # Move to CPU for wavelet and FFT operations
        xy_aug = np.zeros_like(xy_np)

        # Sample mixing coefficient if not provided
        if lambd is None:
            lambd = np.random.beta(0.5, 0.5)

        # Process each sample and channel
        for b in range(batch_size):
            for c in range(enc_in):
                # Wavelet decomposition
                coeffs = pywt.wavedec(
                    xy_np[b, :, c], wavelet=wavelet, level=level, axis=0
                )
                approx = coeffs[0]  # Approximation coefficients
                details = coeffs[1:]  # Detail coefficients

                # Frequency augmentation on detail coefficients
                for i in range(len(details)):
                    freq = fft(details[i])
                    mask = np.random.rand(len(freq)) < mask_rate
                    freq[mask] = 0  # Mask frequencies
                    details[i] = np.real(ifft(freq))

                # Reconstruct augmented series
                max_level = min(level, pywt.dwt_max_level(len(xy_np[b, :, c]), wavelet))
                if max_level < level:
                    coeffs = coeffs[: max_level + 1] + [None] * (level - max_level)
                xy_aug[b, :, c] = pywt.waverec(
                    [approx] + details, wavelet=wavelet, axis=0
                )[:total_len]

        # Mix with original
        xy_aug = lambd * xy_aug + (1 - lambd) * xy_np
        return torch.tensor(xy_aug, dtype=torch.float32)
