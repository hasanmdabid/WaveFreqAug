import torch
import numpy as np
from scipy.signal import savgol_filter  # For STL decomposition
from numpy.fft import fft, ifft


class Augmentation:
    @staticmethod
    def adaptive_channel_preserve(
        x: torch.Tensor,
        y: torch.Tensor,
        segment_size: int = 24,
        mix_rate: float = 0.15,
        scale_factor: float = 0.05,
        variance_threshold: float = 0.03,
        corr_threshold: float = 0.7,
        dim: int = 1,
    ) -> torch.Tensor:
        xy = torch.cat([x, y], dim=dim)
        batch_size, total_len, channels = xy.shape
        xy_mixed = xy.clone()
        corr_matrix = torch.corrcoef(xy[0].permute(1, 0))
        channel_groups = []
        visited = set()
        for c in range(channels):
            if c not in visited:
                group = [c]
                visited.add(c)
                for other_c in range(c + 1, channels):
                    if (
                        corr_matrix[c, other_c] > corr_threshold
                        and other_c not in visited
                    ):
                        group.append(other_c)
                        visited.add(other_c)
                channel_groups.append(group)
        num_segments = (total_len + segment_size - 1) // segment_size
        padded_len = num_segments * segment_size
        if padded_len > total_len:
            padding = torch.zeros(
                batch_size, padded_len - total_len, channels, device=xy.device
            )
            xy_padded = torch.cat([xy, padding], dim=1)
        else:
            xy_padded = xy
        segments = xy_padded[:, : num_segments * segment_size, :].view(
            batch_size, num_segments, segment_size, channels
        )
        variances = segments.var(dim=2)
        temporal_weights = torch.linspace(
            0.1, 1.0, steps=segment_size, device=xy.device
        ).view(1, 1, segment_size, 1)
        b_idx = torch.randperm(batch_size, device=xy.device)
        for group in channel_groups:
            for c in group:
                low_var_mask = variances[:, :, c] < variance_threshold
                for s in range(num_segments):
                    if low_var_mask[:, s].any():
                        segment_start = s * segment_size
                        segment_end = min(segment_start + segment_size, total_len)
                        segment = xy[:, segment_start:segment_end, c : c + 1]
                        segment_mixed = xy[b_idx, segment_start:segment_end, c : c + 1]
                        mix_weights = (
                            torch.rand(segment.shape, device=xy.device)
                            * mix_rate
                            * temporal_weights[:, :, : segment.shape[1], :]
                        )
                        mask = low_var_mask[:, s].view(batch_size, 1, 1)
                        mix_weights = mix_weights * mask
                        xy_mixed[:, segment_start:segment_end, c : c + 1] = (
                            1 - mix_weights
                        ) * segment + mix_weights * segment_mixed
                        scale = torch.rand(
                            (batch_size, 1, 1), device=xy.device, dtype=torch.float
                        ) * scale_factor * 2 + (1 - scale_factor)
                        scale = scale * mask
                        xy_mixed[:, segment_start:segment_end, c : c + 1] *= scale
        xy_mixed = xy_mixed[:, :total_len, :]
        torch.cuda.empty_cache()
        return xy_mixed

    @staticmethod
    def tf_mix(
        x: torch.Tensor,
        y: torch.Tensor,
        segment_size: int = 24,
        mix_rate: float = 0.15,
        scale_factor: float = 0.05,
        variance_threshold: float = 0.03,
        corr_threshold: float = 0.7,
        dim: int = 1,
    ) -> torch.Tensor:
        xy = torch.cat([x, y], dim=dim)
        batch_size, total_len, channels = xy.shape
        xy_mixed = xy.clone()
        corr_matrix = torch.corrcoef(xy[0].permute(1, 0))
        channel_groups = []
        visited = set()
        for c in range(channels):
            if c not in visited:
                group = [c]
                visited.add(c)
                for other_c in range(c + 1, channels):
                    if (
                        corr_matrix[c, other_c] > corr_threshold
                        and other_c not in visited
                    ):
                        group.append(other_c)
                        visited.add(other_c)
                channel_groups.append(group)
        num_segments = (total_len + segment_size - 1) // segment_size
        padded_len = num_segments * segment_size
        if padded_len > total_len:
            padding = torch.zeros(
                batch_size, padded_len - total_len, channels, device=xy.device
            )
            xy_padded = torch.cat([xy, padding], dim=1)
        else:
            xy_padded = xy
        segments = xy_padded[:, : num_segments * segment_size, :].view(
            batch_size, num_segments, segment_size, channels
        )
        variances = segments.var(dim=2)
        b_idx = torch.randperm(batch_size, device=xy.device)
        temporal_weights = torch.linspace(
            0.1, 1.0, steps=segment_size, device=xy.device
        ).view(1, 1, segment_size, 1)
        for group in channel_groups:
            for c in group:
                low_var_mask = variances[:, :, c] < variance_threshold
                for s in range(num_segments):
                    if low_var_mask[:, s].any():
                        segment_start = s * segment_size
                        segment_end = min(segment_start + segment_size, total_len)
                        segment = xy[:, segment_start:segment_end, c : c + 1]
                        segment_mixed = xy[b_idx, segment_start:segment_end, c : c + 1]
                        lam = np.random.beta(mix_rate, mix_rate)
                        mix_weights = (
                            torch.ones(segment.shape, device=xy.device)
                            * lam
                            * temporal_weights[:, :, : segment.shape[1], :]
                        )
                        mask = low_var_mask[:, s].view(batch_size, 1, 1)
                        mix_weights = mix_weights * mask
                        xy_mixed[:, segment_start:segment_end, c : c + 1] = (
                            1 - mix_weights
                        ) * segment + mix_weights * segment_mixed
        xy_f = torch.fft.rfft(xy_mixed, dim=dim)
        amp = torch.abs(xy_f)
        _, index = amp.sort(dim=dim, descending=True)
        dominant_mask = index > 2
        m = (torch.rand(xy_f.shape, device=xy.device) < mix_rate) & dominant_mask
        freal = xy_f.real.masked_fill(m, 0)
        fimag = xy_f.imag.masked_fill(m, 0)
        xy_f_mixed = torch.complex(freal, fimag)
        xy2 = xy[b_idx]
        xy2_f = torch.fft.rfft(xy2, dim=dim)
        freal2 = xy2_f.real.masked_fill(~m, 0)
        fimag2 = xy2_f.imag.masked_fill(~m, 0)
        xy_f_mixed = torch.complex(freal + freal2, fimag + fimag2)
        xy_mixed = torch.fft.irfft(xy_f_mixed, dim=dim)
        for group in channel_groups:
            for c in group:
                scale = torch.rand(
                    (batch_size, 1), device=xy.device
                ) * scale_factor * 2 + (1 - scale_factor)
                xy_mixed[:, :, c] *= scale
        xy_mixed = xy_mixed[:, :total_len, :]
        torch.cuda.empty_cache()
        return xy_mixed

    @staticmethod
    def stl_mix(
        x: torch.Tensor,
        y: torch.Tensor,
        device: torch.device,
        period: int = 7,
        window: int = 15,
        mix_rate: float = 0.15,
        scale_factor: float = 0.05,
        variance_threshold: float = 0.03,
        corr_threshold: float = 0.7,
        dim: int = 1,
    ) -> torch.Tensor:
        def stl_decompose(signal, period, window):
            # Ensure window is odd and at least 3
            window = max(3, window if window % 2 == 1 else window + 1)
            # Trend: Smooth with Savitzky-Golay filter (LOESS approximation)
            trend = savgol_filter(signal, window_length=window, polyorder=2)
            # Detrend
            detrended = signal - trend
            # Seasonal: Average over period
            num_periods = len(signal) // period
            if num_periods < 1:
                seasonal = np.zeros_like(signal)
            else:
                seasonal = np.zeros_like(signal)
                for i in range(num_periods):
                    start = i * period
                    end = min(start + period, len(signal))
                    seasonal[start:end] = np.mean(detrended[start:end])
                # Extend seasonal to full length
                seasonal = np.tile(seasonal[:period], num_periods + 1)[: len(signal)]
            # Remainder
            remainder = signal - trend - seasonal
            return trend, seasonal, remainder

        xy = torch.cat([x, y], dim=dim)
        batch_size, total_len, channels = xy.shape
        xy_mixed = xy.clone()
        corr_matrix = torch.corrcoef(xy[0].permute(1, 0))
        channel_groups = []
        visited = set()
        for c in range(channels):
            if c not in visited:
                group = [c]
                visited.add(c)
                for other_c in range(c + 1, channels):
                    if (
                        corr_matrix[c, other_c] > corr_threshold
                        and other_c not in visited
                    ):
                        group.append(other_c)
                        visited.add(other_c)
                channel_groups.append(group)
        b_idx = torch.randperm(batch_size, device=device)
        for group in channel_groups:
            for c in group:
                xy_c = xy[:, :, c].cpu().numpy()
                for b in range(batch_size):
                    trend, seasonal, remainder = stl_decompose(xy_c[b], period, window)
                    remainder_var = np.var(remainder)
                    if remainder_var < variance_threshold:
                        remainder_other = stl_decompose(
                            xy[b_idx[b], :, c].cpu().numpy(), period, window
                        )[2]
                        lam = np.random.beta(mix_rate, mix_rate)
                        remainder_mixed = (1 - lam) * remainder + lam * remainder_other
                        xy_mixed[b, :, c] = (
                            torch.from_numpy(trend + seasonal + remainder_mixed)
                            .to(device)
                            .float()
                        )
                    else:
                        xy_mixed[b, :, c] = (
                            torch.from_numpy(trend + seasonal + remainder)
                            .to(device)
                            .float()
                        )
                scale = torch.rand(
                    (batch_size, 1), device=device
                ) * scale_factor * 2 + (1 - scale_factor)
                xy_mixed[:, :, c] *= scale
        torch.cuda.empty_cache()
        return xy_mixed

    @staticmethod
    def vmd_mix(
        x: torch.Tensor,
        y: torch.Tensor,
        device: torch.device,
        K: int = 3,
        alpha: int = 2000,
        tau: float = 0,
        mix_rate: float = 0.15,
        scale_factor: float = 0.05,
        variance_threshold: float = 0.03,
        corr_threshold: float = 0.7,
        dim: int = 1,
    ) -> torch.Tensor:
        def vmd(f, alpha, tau, K, DC, init, tol, max_iterations=100):
            N = len(f)
            f_ext = np.concatenate((f[::-1], f, f[::-1]))
            T = len(f_ext)
            t = np.arange(1, T + 1) / T
            if init == 1:
                u_hat = np.zeros((K, T), dtype=complex)
                omega = np.linspace(1, K, K) / (2 * N)
            else:
                u_hat = np.random.randn(K, T) + 1j * np.random.randn(K, T)
                omega = np.sort(np.random.rand(K)) / 2
            lambda_hat = np.zeros(T, dtype=complex)
            u_hat_prev = u_hat.copy()
            n = 0
            while True:
                for k in range(K):
                    other_modes = np.zeros(T, dtype=complex)
                    if k > 0:
                        other_modes += np.sum(u_hat[:k], axis=0)
                    if k < K - 1:
                        other_modes += np.sum(u_hat[k + 1 :], axis=0)
                    u_hat[k] = (f_ext - other_modes + lambda_hat / 2) / (
                        1 + alpha * (t - omega[k]) ** 2
                    )
                for k in range(K):
                    if DC and k == 0:
                        omega[k] = 0
                    else:
                        u_hat_k = u_hat[k]
                        idx = np.where(np.abs(u_hat_k) > tol)[0]
                        if len(idx) > 0:
                            omega[k] = np.sum(
                                t[idx] * np.abs(u_hat_k[idx]) ** 2
                            ) / np.sum(np.abs(u_hat_k[idx]) ** 2)
                lambda_hat = lambda_hat + tau * (f_ext - np.sum(u_hat, axis=0))
                max_change = np.max(np.abs(u_hat - u_hat_prev))
                if max_change < tol:
                    break
                u_hat_prev = u_hat.copy()
                n += 1
                if n > max_iterations:
                    break
            u = np.zeros((K, N))
            for k in range(K):
                u[k] = np.real(ifft(u_hat[k][N : 2 * N]))
            return u, np.abs(u_hat[:, N : 2 * N]), omega

        xy = torch.cat([x, y], dim=dim)
        batch_size, total_len, channels = xy.shape
        xy_mixed = xy.clone()
        corr_matrix = torch.corrcoef(xy[0].permute(1, 0))
        channel_groups = []
        visited = set()
        for c in range(channels):
            if c not in visited:
                group = [c]
                visited.add(c)
                for other_c in range(c + 1, channels):
                    if (
                        corr_matrix[c, other_c] > corr_threshold
                        and other_c not in visited
                    ):
                        group.append(other_c)
                        visited.add(other_c)
                channel_groups.append(group)
        b_idx = torch.randperm(batch_size, device=device)
        for group in channel_groups:
            for c in group:
                xy_c = xy[:, :, c].cpu().numpy()
                for b in range(batch_size):
                    u, _, omega = vmd(
                        xy_c[b], alpha, tau, K, DC=False, init=1, tol=1e-5
                    )
                    u_var = np.var(u, axis=1)
                    low_var_mask = u_var < variance_threshold
                    if np.any(low_var_mask):
                        u_mixed = u.copy()
                        u_other = vmd(
                            xy[b_idx[b], :, c].cpu().numpy(),
                            alpha,
                            tau,
                            K,
                            DC=False,
                            init=1,
                            tol=1e-5,
                        )[0]
                        lam = np.random.beta(mix_rate, mix_rate)
                        u_mixed[low_var_mask] = (1 - lam) * u[
                            low_var_mask
                        ] + lam * u_other[low_var_mask]
                        xy_mixed[b, :, c] = (
                            torch.from_numpy(np.sum(u_mixed, axis=0)).to(device).float()
                        )
                scale = torch.rand(
                    (batch_size, 1), device=device
                ) * scale_factor * 2 + (1 - scale_factor)
                xy_mixed[:, :, c] *= scale
        torch.cuda.empty_cache()
        return xy_mixed
