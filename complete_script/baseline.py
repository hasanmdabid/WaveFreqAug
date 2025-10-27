import torch
import torch.nn as nn
import torch.nn.functional as F
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
        cols_data = df_raw.columns[1:]  # Multivariate, all features
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
                nn.init.xavier_uniform_(self.Linear_Seasonal[i].weight)  # type: ignore
                nn.init.xavier_uniform_(self.Linear_Trend[i].weight)  # type: ignore
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
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](  # type: ignore
                    seasonal_init[:, i, :]
                )
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])  # type: ignore
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
    aug_type,
    seq_len,
    label_len,
    pred_len,
    aug_rate=0.5,
    rates=[0.5, 0.3, 0.1],
    wavelet="db2",
    level=2,
    sampling_rate=0.2,
    n_imf=100,
    epochs=30,
    lr=0.01,
    patience=12,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    aug = Augmentation()
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    path = f"./checkpoints/{aug_type}"
    if not os.path.exists(path):
        os.makedirs(path)

    for epoch in range(epochs):
        model.train()
        train_loss = []
        epoch_time = time.time()
        for i, (batch_x, batch_y, aug_data) in enumerate(train_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            if aug_type == "None":
                pass
            elif aug_type == "Freq-Mask":
                xy = aug.freq_mask(
                    batch_x, batch_y[:, -pred_len:, :], rate=aug_rate, dim=1
                )
                batch_x2, batch_y2 = (
                    xy[:, :seq_len, :],
                    xy[:, seq_len : seq_len + label_len + pred_len, :],
                )
                batch_x = torch.cat([batch_x, batch_x2], dim=0)
                batch_y = torch.cat([batch_y, batch_y2], dim=0)
            elif aug_type == "Freq-Mix":
                xy = aug.freq_mix(
                    batch_x, batch_y[:, -pred_len:, :], rate=aug_rate, dim=1
                )
                batch_x2, batch_y2 = (
                    xy[:, :seq_len, :],
                    xy[:, seq_len : seq_len + label_len + pred_len, :],
                )
                batch_x = torch.cat([batch_x, batch_x2], dim=0)
                batch_y = torch.cat([batch_y, batch_y2], dim=0)
            elif aug_type == "Wave-Mask":
                xy = aug.wave_mask(
                    batch_x,
                    batch_y[:, -pred_len:, :],
                    rates=rates,
                    wavelet=wavelet,
                    level=level,
                    dim=1,
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
            elif aug_type == "Wave-Mix":
                xy = aug.wave_mix(
                    batch_x,
                    batch_y[:, -pred_len:, :],
                    rates=rates,
                    wavelet=wavelet,
                    level=level,
                    dim=1,
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
            elif aug_type == "StAug":
                aug_data = aug_data.float().to(device)
                weighted_xy = aug.emd_aug(aug_data)
                weighted_x, weighted_y = (
                    weighted_xy[:, :seq_len, :],
                    weighted_xy[:, seq_len : seq_len + label_len + pred_len, :],
                )
                batch_x, batch_y = aug.mix_aug(weighted_x, weighted_y, lambd=aug_rate)

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
        "aug_types": [
            "None",
            "Freq-Mask",
            "Freq-Mix",
            "Wave-Mask",
            "Wave-Mix",
            "StAug",
        ],
        "aug_params": {
            96: {
                "None": {
                    "aug_rate": 0.0,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Freq-Mask": {
                    "aug_rate": 0.1,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Freq-Mix": {
                    "aug_rate": 0.2,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Wave-Mask": {
                    "aug_rate": 0.5,
                    "rates": [0.5, 0.3, 0.9, 0.9],
                    "wavelet": "db2",
                    "level": 3,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Wave-Mix": {
                    "aug_rate": 0.5,
                    "rates": [0.0, 0.9, 0.7, 0.7],
                    "wavelet": "db3",
                    "level": 1,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "StAug": {
                    "aug_rate": 0.9,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
            },
            192: {
                "None": {
                    "aug_rate": 0.0,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Freq-Mask": {
                    "aug_rate": 0.5,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Freq-Mix": {
                    "aug_rate": 0.1,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Wave-Mask": {
                    "aug_rate": 0.5,
                    "rates": [0.0, 1.0, 0.2, 1.0],
                    "wavelet": "db3",
                    "level": 1,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Wave-Mix": {
                    "aug_rate": 0.5,
                    "rates": [1.0, 0.4, 0.6, 0.6],
                    "wavelet": "db3",
                    "level": 1,
                    "sampling_rate": 0.8,
                    "n_imf": 100,
                },
                "StAug": {
                    "aug_rate": 0.9,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 900,
                },
            },
            336: {
                "None": {
                    "aug_rate": 0.0,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Freq-Mask": {
                    "aug_rate": 0.5,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Freq-Mix": {
                    "aug_rate": 0.1,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Wave-Mask": {
                    "aug_rate": 0.5,
                    "rates": [0.1, 0.9, 0.4, 0.8],
                    "wavelet": "db25",
                    "level": 1,
                    "sampling_rate": 0.8,
                    "n_imf": 100,
                },
                "Wave-Mix": {
                    "aug_rate": 0.5,
                    "rates": [0.0, 0.9, 0.2, 0.2],
                    "wavelet": "db3",
                    "level": 1,
                    "sampling_rate": 0.8,
                    "n_imf": 100,
                },
                "StAug": {
                    "aug_rate": 0.8,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 200,
                },
            },
            720: {
                "None": {
                    "aug_rate": 0.0,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Freq-Mask": {
                    "aug_rate": 0.4,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Freq-Mix": {
                    "aug_rate": 0.6,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Wave-Mask": {
                    "aug_rate": 0.5,
                    "rates": [0.4, 0.9, 0.5, 1.0],
                    "wavelet": "db1",
                    "level": 1,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Wave-Mix": {
                    "aug_rate": 0.5,
                    "rates": [0.1, 0.9, 0.5, 0.7],
                    "wavelet": "db5",
                    "level": 1,
                    "sampling_rate": 0.8,
                    "n_imf": 100,
                },
                "StAug": {
                    "aug_rate": 0.7,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 1000,
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
        "aug_types": [
            "None",
            "Freq-Mask",
            "Freq-Mix",
            "Wave-Mask",
            "Wave-Mix",
            "StAug",
        ],
        "aug_params": {
            96: {
                "None": {
                    "aug_rate": 0.0,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Freq-Mask": {
                    "aug_rate": 0.6,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Freq-Mix": {
                    "aug_rate": 0.9,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Wave-Mask": {
                    "aug_rate": 0.5,
                    "rates": [0.4, 0.9, 0.0, 0.8],
                    "wavelet": "db26",
                    "level": 2,
                    "sampling_rate": 0.5,
                    "n_imf": 100,
                },
                "Wave-Mix": {
                    "aug_rate": 0.5,
                    "rates": [0.9, 0.9, 0.1, 0.5],
                    "wavelet": "db25",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "StAug": {
                    "aug_rate": 0.4,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 2000,
                },
            },
            192: {
                "None": {
                    "aug_rate": 0.0,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Freq-Mask": {
                    "aug_rate": 0.6,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Freq-Mix": {
                    "aug_rate": 0.8,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Wave-Mask": {
                    "aug_rate": 0.5,
                    "rates": [0.6, 0.7, 0.5, 0.7],
                    "wavelet": "db26",
                    "level": 2,
                    "sampling_rate": 0.8,
                    "n_imf": 100,
                },
                "Wave-Mix": {
                    "aug_rate": 0.5,
                    "rates": [0.9, 0.4, 0.1, 0.8],
                    "wavelet": "db1",
                    "level": 3,
                    "sampling_rate": 0.5,
                    "n_imf": 100,
                },
                "StAug": {
                    "aug_rate": 0.9,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 200,
                },
            },
            336: {
                "None": {
                    "aug_rate": 0.0,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Freq-Mask": {
                    "aug_rate": 0.1,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Freq-Mix": {
                    "aug_rate": 0.8,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Wave-Mask": {
                    "aug_rate": 0.5,
                    "rates": [0.2, 0.7, 0.9, 0.4],
                    "wavelet": "db1",
                    "level": 3,
                    "sampling_rate": 0.8,
                    "n_imf": 100,
                },
                "Wave-Mix": {
                    "aug_rate": 0.5,
                    "rates": [0.9, 0.1, 0.2, 0.5],
                    "wavelet": "db25",
                    "level": 3,
                    "sampling_rate": 0.8,
                    "n_imf": 100,
                },
                "StAug": {
                    "aug_rate": 0.6,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
            },
            720: {
                "None": {
                    "aug_rate": 0.0,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Freq-Mask": {
                    "aug_rate": 0.1,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Freq-Mix": {
                    "aug_rate": 0.1,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Wave-Mask": {
                    "aug_rate": 0.5,
                    "rates": [0.8, 0.9, 0.4, 0.9, 0.5],  # Extended to 5 elements
                    "wavelet": "db5",
                    "level": 4,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Wave-Mix": {
                    "aug_rate": 0.5,
                    "rates": [0.5, 0.1, 0.2, 0.7],
                    "wavelet": "db5",
                    "level": 1,
                    "sampling_rate": 1.0,
                    "n_imf": 100,
                },
                "StAug": {
                    "aug_rate": 0.4,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 700,
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
        "aug_types": [
            "None",
            "Freq-Mask",
            "Freq-Mix",
            "Wave-Mask",
            "Wave-Mix",
        ],
        "aug_params": {
            96: {
                "None": {
                    "aug_rate": 0.0,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Freq-Mask": {
                    "aug_rate": 0.1,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Freq-Mix": {
                    "aug_rate": 0.9,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Wave-Mask": {
                    "aug_rate": 0.5,
                    "rates": [0.2, 1.0, 0.4, 0.4],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.5,
                    "n_imf": 100,
                },
                "Wave-Mix": {
                    "aug_rate": 0.5,
                    "rates": [0.1, 0.5, 0.1, 0.2],
                    "wavelet": "db3",
                    "level": 1,
                    "sampling_rate": 1.0,
                    "n_imf": 100,
                },
            },
            192: {
                "None": {
                    "aug_rate": 0.0,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Freq-Mask": {
                    "aug_rate": 0.1,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Freq-Mix": {
                    "aug_rate": 0.1,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Wave-Mask": {
                    "aug_rate": 0.5,
                    "rates": [0.1, 0.7, 0.1, 0.4],
                    "wavelet": "db2",
                    "level": 1,
                    "sampling_rate": 0.5,
                    "n_imf": 100,
                },
                "Wave-Mix": {
                    "aug_rate": 0.5,
                    "rates": [0.2, 0.7, 1.0, 0.3],
                    "wavelet": "db3",
                    "level": 1,
                    "sampling_rate": 1.0,
                    "n_imf": 100,
                },
            },
            336: {
                "None": {
                    "aug_rate": 0.0,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Freq-Mask": {
                    "aug_rate": 0.1,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Freq-Mix": {
                    "aug_rate": 0.1,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Wave-Mask": {
                    "aug_rate": 0.5,
                    "rates": [1.0, 1.0, 0.0, 0.0],
                    "wavelet": "db1",
                    "level": 1,
                    "sampling_rate": 1.0,
                    "n_imf": 100,
                },
                "Wave-Mix": {
                    "aug_rate": 0.5,
                    "rates": [0.8, 0.6, 0.8, 0.6],
                    "wavelet": "db2",
                    "level": 1,
                    "sampling_rate": 1.0,
                    "n_imf": 100,
                },
            },
            720: {
                "None": {
                    "aug_rate": 0.0,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Freq-Mask": {
                    "aug_rate": 0.9,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Freq-Mix": {
                    "aug_rate": 0.9,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Wave-Mask": {
                    "aug_rate": 0.5,
                    "rates": [1.0, 0.8, 0.6, 0.0],
                    "wavelet": "db2",
                    "level": 1,
                    "sampling_rate": 0.5,
                    "n_imf": 100,
                },
                "Wave-Mix": {
                    "aug_rate": 0.5,
                    "rates": [0.1, 0.1, 0.7, 0.5],
                    "wavelet": "db1",
                    "level": 1,
                    "sampling_rate": 1.0,
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
        "aug_types": [
            "None",
            "Freq-Mask",
            "Freq-Mix",
            "Wave-Mask",
            "Wave-Mix",
            "StAug",
        ],
        "aug_params": {
            24: {
                "None": {
                    "aug_rate": 0.0,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Freq-Mask": {
                    "aug_rate": 0.2,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Freq-Mix": {
                    "aug_rate": 0.1,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Wave-Mask": {
                    "aug_rate": 0.5,
                    "rates": [0.4, 0.8, 0.9, 0.7],
                    "wavelet": "db25",
                    "level": 1,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Wave-Mix": {
                    "aug_rate": 0.5,
                    "rates": [0.1, 0.8, 1.0, 0.0],
                    "wavelet": "db1",
                    "level": 1,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "StAug": {
                    "aug_rate": 0.7,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 200,
                },
            },
            36: {
                "None": {
                    "aug_rate": 0.0,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Freq-Mask": {
                    "aug_rate": 0.1,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Freq-Mix": {
                    "aug_rate": 0.1,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Wave-Mask": {
                    "aug_rate": 0.5,
                    "rates": [0.6, 0.8, 0.3, 0.1],
                    "wavelet": "db25",
                    "level": 1,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Wave-Mix": {
                    "aug_rate": 0.5,
                    "rates": [0.1, 1.0, 0.9, 0.2],
                    "wavelet": "db25",
                    "level": 1,
                    "sampling_rate": 0.8,
                    "n_imf": 100,
                },
                "StAug": {
                    "aug_rate": 0.3,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 300,
                },
            },
            48: {
                "None": {
                    "aug_rate": 0.0,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Freq-Mask": {
                    "aug_rate": 0.1,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Freq-Mix": {
                    "aug_rate": 0.1,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Wave-Mask": {
                    "aug_rate": 0.5,
                    "rates": [0.2, 0.7, 1.0, 0.4],
                    "wavelet": "db2",
                    "level": 1,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Wave-Mix": {
                    "aug_rate": 0.5,
                    "rates": [0.1, 1.0, 0.4, 0.5],
                    "wavelet": "db3",
                    "level": 1,
                    "sampling_rate": 1.0,
                    "n_imf": 100,
                },
                "StAug": {
                    "aug_rate": 0.9,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 300,
                },
            },
            60: {
                "None": {
                    "aug_rate": 0.0,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Freq-Mask": {
                    "aug_rate": 0.1,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Freq-Mix": {
                    "aug_rate": 0.1,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Wave-Mask": {
                    "aug_rate": 0.5,
                    "rates": [0.2, 0.8, 0.5, 0.1],
                    "wavelet": "db25",
                    "level": 1,
                    "sampling_rate": 0.2,
                    "n_imf": 100,
                },
                "Wave-Mix": {
                    "aug_rate": 0.5,
                    "rates": [0.1, 0.9, 0.3, 0.9],
                    "wavelet": "db1",
                    "level": 1,
                    "sampling_rate": 0.5,
                    "n_imf": 100,
                },
                "StAug": {
                    "aug_rate": 0.7,
                    "rates": [0.5, 0.3, 0.1],
                    "wavelet": "db2",
                    "level": 2,
                    "sampling_rate": 0.2,
                    "n_imf": 1000,
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
    patience = 10
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

            # Run experiments for each augmentation type
            for aug_type in config["aug_types"]:
                params = config["aug_params"][pred_len][aug_type]
                mse_list, mae_list, rse_list = [], [], []
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
                        aug_type,
                        config["seq_len"],
                        label_len,
                        pred_len,
                        aug_rate=params["aug_rate"],
                        rates=params["rates"],
                        wavelet=params["wavelet"],
                        level=params["level"],
                        sampling_rate=params["sampling_rate"],
                        n_imf=params["n_imf"],
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
                    print(
                        f"Iteration {itr+1} - Val Loss: {val_loss:.6f}, MAE: {mae:.6f}, MSE: {mse:.6f}, RSE: {rse:.6f}"
                    )

                results[(pred_len, aug_type)] = {
                    "val_loss": np.mean(
                        [
                            train(
                                model,
                                train_loader,
                                val_loader,
                                device,
                                aug_type,
                                config["seq_len"],
                                label_len,
                                pred_len,
                                aug_rate=params["aug_rate"],
                                rates=params["rates"],
                                wavelet=params["wavelet"],
                                level=params["level"],
                                sampling_rate=params["sampling_rate"],
                                n_imf=params["n_imf"],
                                epochs=epochs,
                                lr=learning_rate,
                                patience=patience,
                            )
                            for _ in range(num_iterations)
                        ]
                    ),
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
        with open(f"results_{dataset_name}.txt", "w") as f:
            for pred_len in config["pred_lens"]:
                for aug_type in config["aug_types"]:
                    metrics = results[(pred_len, aug_type)]
                    f.write(
                        f"{dataset_name}, pred_len={pred_len}, {aug_type}: "
                        f'Val Loss={metrics["val_loss"]:.6f}, MAE={metrics["mae"]:.6f}, MSE={metrics["mse"]:.6f}, '
                        f'RSE={metrics["rse"]:.6f}, MAE Std={metrics["mae_std"]:.6f}, MSE Std={metrics["mse_std"]:.6f}, '
                        f'RSE Std={metrics["rse_std"]:.6f}\n'
                    )
