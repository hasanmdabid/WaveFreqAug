import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pytorch_wavelets import DWT1DForward, DWT1DInverse
from PyEMD import EMD
import os
import random
import time
import json

# Set random seed for reproducibility
fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


# Early Stopping class
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ..."
            )
        torch.save(model.state_dict(), os.path.join(path, "checkpoint.pth"))
        self.val_loss_min = val_loss

    def get_val_loss_min(self):
        return self.val_loss_min


# Learning rate adjustment
def adjust_learning_rate(optimizer, epoch, lr):
    lr_adjust = {epoch: lr * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust:
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print(f"Updating learning rate to {lr}")


# Augmentation class
class Augmentation:
    @staticmethod
    def wavelet_adaptive_mix(
        x: torch.Tensor,
        y: torch.Tensor,
        mix_rate: float = 0.3,
        wavelet: str = "db2",
        level: int = 2,
        dim: int = 1,
        mix_k: int = 5,
        perturb_scale: float = 0.01,
    ) -> torch.Tensor:
        """
        Wavelet-Adaptive-Mix: Mixes wavelet coefficients adaptively based on their energy.

        Args:
            x: Input tensor (batch_size, seq_len, channels)
            y: Target tensor (batch_size, pred_len, channels)
            mix_rate: Maximum mixing weight for coefficients
            wavelet: Wavelet type for DWT
            level: Number of decomposition levels
            dim: Dimension to apply DWT
            mix_k: Number of coefficients to mix
            perturb_scale: Scale of random perturbation for non-mixed coefficients

        Returns:
            Augmented tensor (batch_size, seq_len + pred_len, channels)
        """
        xy = torch.cat([x, y], dim=dim)
        batch_size, total_len, channels = xy.shape
        xy = xy.permute(0, 2, 1).to(x.device).to(torch.float64)

        # Dynamically adjust mix_k
        mix_k = min(mix_k, total_len // (2**level))

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
        cA_abs = torch.abs(cA)
        cDs_abs = [torch.abs(cD) for cD in cDs]

        # Compute energy-based weights for mixing
        cA_energy = cA_abs / (cA_abs.sum(dim=-1, keepdim=True) + 1e-8)
        cDs_energy = [
            cD_abs / (cD_abs.sum(dim=-1, keepdim=True) + 1e-8) for cD_abs in cDs_abs
        ]

        # Select top mix_k coefficients for mixing
        mask_cA = torch.zeros_like(cA, dtype=torch.bool)
        for b in range(batch_size):
            for c in range(channels):
                _, indices = cA_abs[b, c].sort(descending=True)
                mask_cA[b, c, indices[:mix_k]] = True

        masked_cDs = []
        for cD in cDs_abs:
            mask_cD = torch.zeros_like(cD, dtype=torch.bool)
            for b in range(batch_size):
                for c in range(channels):
                    _, indices = cD[b, c].sort(descending=True)
                    mask_cD[b, c, indices[:mix_k]] = True
            masked_cDs.append(mask_cD)

        # Generate mixing weights
        b_idx = torch.randperm(batch_size)
        mix_weights = (
            torch.rand(cA.shape, device=cA.device, dtype=torch.float) * mix_rate
        )
        mix_weights = (
            mix_weights * mask_cA.float()
        )  # Apply only to selected coefficients
        mix_weights = mix_weights * cA_energy  # Weight by energy

        cA_mixed = cA.clone()
        cA_mixed[mask_cA] = (1 - mix_weights[mask_cA]) * cA[mask_cA] + mix_weights[
            mask_cA
        ] * cA[b_idx][mask_cA]

        cDs_mixed = []
        for cD, mask, energy in zip(cDs, masked_cDs, cDs_energy):
            cD_mixed = cD.clone()
            mix_weights = (
                torch.rand(cD.shape, device=cD.device, dtype=torch.float) * mix_rate
            )
            mix_weights = mix_weights * mask.float() * energy
            cD_mixed[mask] = (1 - mix_weights[mask]) * cD[mask] + mix_weights[
                mask
            ] * cD[b_idx][mask]
            cDs_mixed.append(cD_mixed)

        # Apply small perturbation to non-mixed coefficients
        cA_mixed[~mask_cA] += (
            torch.randn_like(cA_mixed[~mask_cA], dtype=torch.float64) * perturb_scale
        )
        for i, (cD, mask) in enumerate(zip(cDs_mixed, masked_cDs)):
            cDs_mixed[i][~mask] += (
                torch.randn_like(cDs_mixed[i][~mask], dtype=torch.float64)
                * perturb_scale
            )

        try:
            idwt = DWT1DInverse(wave=wavelet, mode="symmetric").to(x.device).double()
        except:
            idwt = DWT1DInverse(wave="db1", mode="symmetric").to(x.device).double()

        reconstructed = idwt((cA_mixed, cDs_mixed))
        reconstructed = reconstructed.permute(0, 2, 1)
        return reconstructed.float()


# EMD decomposition
def emd_augment(data, sequence_length, n_IMF=500):
    n_imf, channel_num = n_IMF, data.shape[1]
    emd_data = np.zeros((n_imf, data.shape[0], channel_num))
    max_imf = 0
    for ci in range(channel_num):
        s = data[:, ci]
        IMF = EMD().emd(s)
        r_s = np.zeros((n_imf, data.shape[0]))
        if len(IMF) > max_imf:
            max_imf = len(IMF)
        for i in range(len(IMF)):
            r_s[i] = IMF[len(IMF) - 1 - i]
        if len(IMF) == 0:
            r_s[0] = s
        emd_data[:, :, ci] = r_s
    if max_imf < n_imf:
        emd_data = emd_data[:max_imf, :, :]
    train_data_new = np.zeros(
        (len(data) - sequence_length + 1, max_imf, sequence_length, channel_num)
    )
    for i in range(len(data) - sequence_length + 1):
        train_data_new[i] = emd_data[:, i : i + sequence_length, :]
    return train_data_new


# Dataset class
class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        data_path,
        data_name,
        flag="train",
        seq_len=336,
        label_len=0,
        pred_len=96,
        enc_in=7,
        n_imf=500,
    ):
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.n_imf = n_imf
        self.data_name = data_name
        assert flag in ["train", "val", "test"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(self.data_path)
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
        self.scaler = StandardScaler()

        if self.data_name in ["ETTh1", "ETTh2"]:
            border1s = [
                0,
                12 * 30 * 24 - self.seq_len,
                12 * 30 * 24 + 4 * 30 * 24 - self.seq_len,
            ]
            border2s = [
                12 * 30 * 24,
                12 * 30 * 24 + 4 * 30 * 24,
                12 * 30 * 24 + 8 * 30 * 24,
            ]
        elif self.data_name in ["weather", "custom"]:
            border1s = [
                0,
                int(0.6 * len(df_raw)) - self.seq_len,
                int(0.8 * len(df_raw)) - self.seq_len,
            ]
            border2s = [int(0.6 * len(df_raw)), int(0.8 * len(df_raw)), len(df_raw)]
        else:
            raise ValueError(f"Unsupported data_name: {self.data_name}")

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        train_data = df_data[border1s[0] : border2s[0]]
        self.scaler.fit(train_data.values)
        data = self.scaler.transform(df_data.values)
        if self.set_type == 0:  # Train
            self.aug_data = emd_augment(
                data[border1:border2], self.seq_len + self.pred_len, n_IMF=self.n_imf
            )
            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]
        else:
            self.aug_data = np.zeros(
                (len(data[border1:border2]) - self.seq_len - self.pred_len + 1, 0, 0, 0)
            )
            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        aug_data = self.aug_data[s_begin] if self.set_type == 0 else np.array([])
        return seq_x, seq_y, aug_data

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# DLinear model
class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, individual=False):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.individual = individual
        kernel_size = 25
        self.decomp = SeriesDecomp(kernel_size)
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
                nn.init.xavier_uniform_(self.Linear_Seasonal[i].weight) # type: ignore
                nn.init.xavier_uniform_(self.Linear_Trend[i].weight) # type: ignore
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
            nn.init.xavier_uniform_(self.Linear_Seasonal.weight)
            nn.init.xavier_uniform_(self.Linear_Trend.weight)

    def forward(self, x):
        seasonal_init, trend_init = self.decomp(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(
            0, 2, 1
        )
        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                dtype=seasonal_init.dtype,
            ).to(seasonal_init.device)
            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), self.pred_len],
                dtype=trend_init.dtype,
            ).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i]( # type: ignore
                    seasonal_init[:, i, :]
                )
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :]) # type: ignore
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)


# Metrics
def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(
        np.sum((true - true.mean()) ** 2)
    )


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def train(
    model,
    train_loader,
    val_loader,
    device,
    seq_len,
    label_len,
    pred_len,
    mix_rate=0.3,
    wavelet="db2",
    level=2,
    mix_k=5,
    perturb_scale=0.01,
    sampling_rate=0.5,
    epochs=30,
    lr=0.01,
    patience=12,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    aug = Augmentation()
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    path = "./checkpoints/Wavelet-Adaptive-Mix"
    if not os.path.exists(path):
        os.makedirs(path)

    for epoch in range(epochs):
        model.train()
        train_loss = []
        epoch_time = time.time()
        for i, (batch_x, batch_y, aug_data) in enumerate(train_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            xy = aug.wavelet_adaptive_mix(
                batch_x,
                batch_y[:, -pred_len:, :],
                mix_rate=mix_rate,
                wavelet=wavelet,
                level=level,
                dim=1,
                mix_k=mix_k,
                perturb_scale=perturb_scale,
            )
            batch_x2, batch_y2 = (
                xy[:, :seq_len, :],
                xy[:, seq_len : seq_len + label_len + pred_len, :],
            )
            sampling_steps = int(batch_x2.shape[0] * sampling_rate)
            indices = torch.randperm(batch_x2.shape[0])[:sampling_steps]
            batch_x2, batch_y2 = batch_x2[indices, :, :], batch_y2[indices, :, :]
            batch_x = torch.cat([batch_x, batch_x2], dim=0)
            batch_y = torch.cat([batch_y, batch_y2], dim=0)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs[:, -pred_len:, :], batch_y[:, -pred_len:, :])
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        train_loss = np.average(train_loss)
        val_loss = validate(model, val_loader, device, criterion, pred_len)
        print(
            f"Epoch: {epoch + 1}, Time: {time.time() - epoch_time:.2f}s | Train Loss: {train_loss:.7f} Val Loss: {val_loss:.7f}"
        )

        early_stopping(val_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        adjust_learning_rate(optimizer, epoch + 1, lr)

    return early_stopping.get_val_loss_min()


def validate(model, val_loader, device, criterion, pred_len):
    model.eval()
    total_loss = []
    with torch.no_grad():
        for batch_x, batch_y, _ in val_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            outputs = model(batch_x)
            loss = criterion(outputs[:, -pred_len:, :], batch_y[:, -pred_len:, :])
            total_loss.append(loss.item())
    return np.average(total_loss)


def test(model, test_loader, device, scaler, pred_len):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch_x, batch_y, _ in test_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            outputs = model(batch_x)
            outputs = outputs[:, -pred_len:, :].cpu().numpy()
            batch_y = batch_y[:, -pred_len:, :].cpu().numpy()
            preds.append(outputs)
            trues.append(batch_y)

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    mae = MAE(preds, trues)
    mse = MSE(preds, trues)
    rse = RSE(preds, trues)
    return mae, mse, rse


# Dataset configurations
dataset_configs = {
    "ETTh1": {
        "data_path": "./dataset/ETTh1.csv",
        "data_name": "ETTh1",
        "seq_len": 336,
        "enc_in": 7,
        "batch_size": 64,
        "pred_lens": [96, 192, 336, 720],
        "aug_types": ["Wavelet-Adaptive-Mix"],
        "aug_params": {
            96: {
                "Wavelet-Adaptive-Mix": {
                    "mix_rate": 0.3,
                    "wavelet": "db2",
                    "level": 2,
                    "mix_k": 5,
                    "perturb_scale": 0.01,
                    "sampling_rate": 0.5,
                    "n_imf": 100,
                },
            },
            192: {
                "Wavelet-Adaptive-Mix": {
                    "mix_rate": 0.3,
                    "wavelet": "db3",
                    "level": 2,
                    "mix_k": 5,
                    "perturb_scale": 0.01,
                    "sampling_rate": 0.5,
                    "n_imf": 100,
                },
            },
            336: {
                "Wavelet-Adaptive-Mix": {
                    "mix_rate": 0.3,
                    "wavelet": "sym4",
                    "level": 2,
                    "mix_k": 10,
                    "perturb_scale": 0.01,
                    "sampling_rate": 0.8,
                    "n_imf": 100,
                },
            },
            720: {
                "Wavelet-Adaptive-Mix": {
                    "mix_rate": 0.3,
                    "wavelet": "db5",
                    "level": 3,
                    "mix_k": 10,
                    "perturb_scale": 0.01,
                    "sampling_rate": 0.5,
                    "n_imf": 100,
                },
            },
        },
    },
    "ETTh2": {
        "data_path": "./dataset/ETTh2.csv",
        "data_name": "ETTh2",
        "seq_len": 336,
        "enc_in": 7,
        "batch_size": 64,
        "pred_lens": [96, 192, 336, 720],
        "aug_types": ["Wavelet-Adaptive-Mix"],
        "aug_params": {
            96: {
                "Wavelet-Adaptive-Mix": {
                    "mix_rate": 0.3,
                    "wavelet": "sym4",
                    "level": 2,
                    "mix_k": 5,
                    "perturb_scale": 0.01,
                    "sampling_rate": 0.5,
                    "n_imf": 100,
                },
            },
            192: {
                "Wavelet-Adaptive-Mix": {
                    "mix_rate": 0.3,
                    "wavelet": "db1",
                    "level": 2,
                    "mix_k": 5,
                    "perturb_scale": 0.01,
                    "sampling_rate": 0.8,
                    "n_imf": 100,
                },
            },
            336: {
                "Wavelet-Adaptive-Mix": {
                    "mix_rate": 0.3,
                    "wavelet": "db1",
                    "level": 2,
                    "mix_k": 10,
                    "perturb_scale": 0.01,
                    "sampling_rate": 0.8,
                    "n_imf": 100,
                },
            },
            720: {
                "Wavelet-Adaptive-Mix": {
                    "mix_rate": 0.3,
                    "wavelet": "db5",
                    "level": 3,
                    "mix_k": 10,
                    "perturb_scale": 0.01,
                    "sampling_rate": 0.5,
                    "n_imf": 100,
                },
            },
        },
    },
    "weather": {
        "data_path": "./dataset/weather.csv",
        "data_name": "weather",
        "seq_len": 336,
        "enc_in": 21,
        "batch_size": 64,
        "pred_lens": [96, 192, 336, 720],
        "aug_types": ["Wavelet-Adaptive-Mix"],
        "aug_params": {
            96: {
                "Wavelet-Adaptive-Mix": {
                    "mix_rate": 0.3,
                    "wavelet": "db2",
                    "level": 2,
                    "mix_k": 5,
                    "perturb_scale": 0.01,
                    "sampling_rate": 0.5,
                    "n_imf": 100,
                },
            },
            192: {
                "Wavelet-Adaptive-Mix": {
                    "mix_rate": 0.3,
                    "wavelet": "db3",
                    "level": 1,
                    "mix_k": 5,
                    "perturb_scale": 0.01,
                    "sampling_rate": 1.0,
                    "n_imf": 100,
                },
            },
            336: {
                "Wavelet-Adaptive-Mix": {
                    "mix_rate": 0.3,
                    "wavelet": "db1",
                    "level": 1,
                    "mix_k": 10,
                    "perturb_scale": 0.01,
                    "sampling_rate": 1.0,
                    "n_imf": 100,
                },
            },
            720: {
                "Wavelet-Adaptive-Mix": {
                    "mix_rate": 0.3,
                    "wavelet": "db2",
                    "level": 2,
                    "mix_k": 10,
                    "perturb_scale": 0.01,
                    "sampling_rate": 0.5,
                    "n_imf": 100,
                },
            },
        },
    },
    "ILI": {
        "data_path": "./dataset/national_illness.csv",
        "data_name": "custom",
        "seq_len": 36,
        "enc_in": 7,
        "batch_size": 32,
        "pred_lens": [24, 36, 48, 60],
        "aug_types": ["Wavelet-Adaptive-Mix"],
        "aug_params": {
            24: {
                "Wavelet-Adaptive-Mix": {
                    "mix_rate": 0.2,
                    "wavelet": "db25",
                    "level": 1,
                    "mix_k": 3,
                    "perturb_scale": 0.005,
                    "sampling_rate": 0.5,
                    "n_imf": 100,
                },
            },
            36: {
                "Wavelet-Adaptive-Mix": {
                    "mix_rate": 0.2,
                    "wavelet": "db25",
                    "level": 1,
                    "mix_k": 3,
                    "perturb_scale": 0.005,
                    "sampling_rate": 0.5,
                    "n_imf": 100,
                },
            },
            48: {
                "Wavelet-Adaptive-Mix": {
                    "mix_rate": 0.2,
                    "wavelet": "db2",
                    "level": 1,
                    "mix_k": 3,
                    "perturb_scale": 0.005,
                    "sampling_rate": 0.5,
                    "n_imf": 100,
                },
            },
            60: {
                "Wavelet-Adaptive-Mix": {
                    "mix_rate": 0.2,
                    "wavelet": "db25",
                    "level": 1,
                    "mix_k": 3,
                    "perturb_scale": 0.005,
                    "sampling_rate": 0.5,
                    "n_imf": 100,
                },
            },
        },
    },
}

# Main experiment
if __name__ == "__main__":
    # Common parameters
    epochs = 30
    learning_rate = 0.01
    patience = 12
    num_iterations = 10
    label_len = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create directories
    if not os.path.exists("./dataset"):
        os.makedirs("./dataset")
    if not os.path.exists("./checkpoints"):
        os.makedirs("./checkpoints")
    if not os.path.exists("./plots"):
        os.makedirs("./plots")

    # Run experiments for each dataset
    for dataset_name, config in dataset_configs.items():
        print(f"\n=== Processing dataset: {dataset_name} ===")
        results = {}
        # Load datasets
        for pred_len in config["pred_lens"]:
            print(f"\nPrediction length: {pred_len}")
            train_dataset = TimeSeriesDataset(
                data_path=config["data_path"],
                data_name=config["data_name"],
                flag="train",
                seq_len=config["seq_len"],
                label_len=label_len,
                pred_len=pred_len,
                enc_in=config["enc_in"],
                n_imf=max(
                    [
                        config["aug_params"][pred_len][aug_type]["n_imf"]
                        for aug_type in config["aug_types"]
                    ]
                ),
            )
            val_dataset = TimeSeriesDataset(
                data_path=config["data_path"],
                data_name=config["data_name"],
                flag="val",
                seq_len=config["seq_len"],
                label_len=label_len,
                pred_len=pred_len,
                enc_in=config["enc_in"],
            )
            test_dataset = TimeSeriesDataset(
                data_path=config["data_path"],
                data_name=config["data_name"],
                flag="test",
                seq_len=config["seq_len"],
                label_len=label_len,
                pred_len=pred_len,
                enc_in=config["enc_in"],
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=config["batch_size"],
                shuffle=True,
                drop_last=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config["batch_size"],
                shuffle=False,
                drop_last=True,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=config["batch_size"],
                shuffle=False,
                drop_last=True,
            )

            # Run experiment for Wavelet-Adaptive-Mix
            aug_type = "Wavelet-Adaptive-Mix"
            params = config["aug_params"][pred_len][aug_type]
            mse_list, mae_list, rse_list, val_loss_list = [], [], [], []
            print(
                f"\nRunning experiment with {aug_type} augmentation for {dataset_name}, pred_len={pred_len}..."
            )
            for itr in range(num_iterations):
                print(f"Iteration {itr+1}/{num_iterations}")
                model = DLinear(
                    config["seq_len"],
                    pred_len,
                    enc_in=config["enc_in"],
                    individual=False,
                ).to(device)
                val_loss = train(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    config["seq_len"],
                    label_len,
                    pred_len,
                    mix_rate=params["mix_rate"],
                    wavelet=params["wavelet"],
                    level=params["level"],
                    mix_k=params["mix_k"],
                    perturb_scale=params["perturb_scale"],
                    sampling_rate=params["sampling_rate"],
                    epochs=epochs,
                    lr=learning_rate,
                    patience=patience,
                )
                model.load_state_dict(
                    torch.load(f"./checkpoints/{aug_type}/checkpoint.pth")
                )
                mae, mse, rse = test(
                    model, test_loader, device, train_dataset.scaler, pred_len
                )
                mse_list.append(mse)
                mae_list.append(mae)
                rse_list.append(rse)
                val_loss_list.append(val_loss)
                print(
                    f"Iteration {itr+1} - Val Loss: {val_loss:.6f}, MAE: {mae:.6f}, MSE: {mse:.6f}, RSE: {rse:.6f}"
                )

            results[(pred_len, aug_type)] = {
                "val_loss": np.mean(val_loss_list),
                "mae": np.mean(mae_list),
                "mse": np.mean(mse_list),
                "rse": np.mean(rse_list),
                "mae_std": np.std(mae_list),
                "mse_std": np.std(mse_list),
                "rse_std": np.std(rse_list),
            }
            print(
                f'{aug_type} - Avg Val Loss: {results[(pred_len, aug_type)]["val_loss"]:.6f}, Avg MAE: {results[(pred_len, aug_type)]["mae"]:.6f}, '
                f'Avg MSE: {results[(pred_len, aug_type)]["mse"]:.6f}, Avg RSE: {results[(pred_len, aug_type)]["rse"]:.6f}, '
                f'MSE Std: {results[(pred_len, aug_type)]["mse_std"]:.6f}'
            )

            # Plot predictions
            model.load_state_dict(
                torch.load(f"./checkpoints/{aug_type}/checkpoint.pth")
            )
            model.eval()
            with torch.no_grad():
                batch_x, batch_y, _ = next(iter(test_loader))
                batch_x = batch_x.float().to(device)
                outputs = model(batch_x)
                outputs = outputs[:, -pred_len:, :].cpu().numpy()
                batch_y = batch_y[:, -pred_len:, :].cpu().numpy()
                outputs = train_dataset.scaler.inverse_transform(outputs[0])
                batch_y = train_dataset.scaler.inverse_transform(batch_y[0])
                plt.figure()
                plt.plot(batch_y[:, 0], label="Ground Truth")
                plt.plot(outputs[:, 0], label=f"Prediction ({aug_type})")
                plt.legend()
                plt.savefig(
                    f"./plots/prediction_{dataset_name}_{aug_type}_{pred_len}.png"
                )
                plt.close()

        # Save results
        with open(f"results_{dataset_name}_wavelet_adaptive_mix.txt", "w") as f:
            for pred_len in config["pred_lens"]:
                for aug_type in config["aug_types"]:
                    metrics = results[(pred_len, aug_type)]
                    f.write(
                        f"{dataset_name}, pred_len={pred_len}, {aug_type}: "
                        f'Val Loss={metrics["val_loss"]:.6f}, MAE={metrics["mae"]:.6f}, MSE={metrics["mse"]:.6f}, '
                        f'RSE={metrics["rse"]:.6f}, MAE Std={metrics["mae_std"]:.6f}, MSE Std={metrics["mse_std"]:.6f}, '
                        f'RSE Std={metrics["rse_std"]:.6f}\n'
                    )
