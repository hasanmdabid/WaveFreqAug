import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import random
import time


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
        print(f"Adjusted learning rate to {lr}")


# Augmentation class
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
        """
        Adaptive-Channel-Preserve: Mixes low-variance segments within correlated channel groups, preserves high-variance segments,
        and applies subtle scaling to low-variance segments.

        Args:
            x: Input tensor (batch_size, seq_len, channels)
            y: Target tensor (batch_size, pred_len, channels)
            segment_size: Size of each segment for variance analysis
            mix_rate: Maximum mixing weight for low-variance segments
            scale_factor: Scaling range for low-variance segments (e.g., [1-scale_factor, scale_factor + 1])
            variance_threshold: Threshold to identify low-variance segments
            corr_threshold: Correlation threshold for channel grouping
            dim: Time dimension for concatenation

        Returns:
            Augmented tensor (batch_size, seq_len + pred_len, channels)
        """
        xy = torch.cat([x, y], dim=dim)
        batch_size, total_len, channels = xy.shape

        # Initialize output tensor
        xy_mixed = xy.clone()

        # Compute channel correlations (using first batch sample for consistency)
        corr_matrix = torch.corrcoef(xy[0].permute(1, 0))  # Shape: (channels, channels)

        # Group channels based on correlation
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

        # Divide sequence into non-overlapping segments
        num_segments = (
            total_len + segment_size - 1
        ) // segment_size  # Ceiling division
        padded_len = num_segments * segment_size
        if padded_len > total_len:
            # Pad xy to match padded_len
            padding = torch.zeros(
                batch_size, padded_len - total_len, channels, device=xy.device
            )
            xy_padded = torch.cat([xy, padding], dim=1)
        else:
            xy_padded = xy

        # Reshape into segments
        segments = xy_padded[:, : num_segments * segment_size, :].view(
            batch_size, num_segments, segment_size, channels
        )

        # Compute variance for each segment
        variances = segments.var(dim=2)  # Shape: (batch_size, num_segments, channels)

        # Temporal weights for mixing
        temporal_weights = torch.linspace(
            0.1, 1.0, steps=segment_size, device=xy.device
        ).view(1, 1, segment_size, 1)

        # Process each channel group
        b_idx = torch.randperm(batch_size)
        for group in channel_groups:
            for c in group:
                low_var_mask = (
                    variances[:, :, c] < variance_threshold
                )  # Shape: (batch_size, num_segments)
                for s in range(num_segments):
                    if low_var_mask[:, s].any():
                        # Extract segment
                        segment_start = s * segment_size
                        segment_end = min(segment_start + segment_size, total_len)
                        segment = xy[
                            :, segment_start:segment_end, c : c + 1
                        ]  # Shape: (batch_size, segment_size, 1)
                        segment_mixed = xy[
                            b_idx, segment_start:segment_end, c : c + 1
                        ]  # Shape: (batch_size, segment_size, 1)

                        # Apply mixing weights
                        mix_weights = (
                            torch.rand(
                                segment.shape, device=xy.device
                            )  # Shape: (batch_size, segment_size, 1)
                            * mix_rate
                            * temporal_weights[
                                :, :, : segment.shape[1], :
                            ]  # Shape: (1, 1, segment_size, 1)
                        )
                        # Apply low-variance mask correctly
                        mask = low_var_mask[:, s].view(
                            batch_size, 1, 1
                        )  # Shape: (batch_size, 1, 1)
                        mix_weights = (
                            mix_weights * mask
                        )  # Shape: (batch_size, segment_size, 1)

                        xy_mixed[:, segment_start:segment_end, c : c + 1] = (
                            1 - mix_weights
                        ) * segment + mix_weights * segment_mixed

                        # Apply scaling to low-variance segments
                        scale = torch.rand(
                            (batch_size, 1, 1), device=xy.device, dtype=torch.float
                        ) * scale_factor * 2 + (
                            1 - scale_factor
                        )  # Shape: (batch_size, 1, 1)
                        scale = scale * mask  # Shape: (batch_size, 1, 1)
                        xy_mixed[:, segment_start:segment_end, c : c + 1] *= scale

        return xy_mixed[:, :total_len, :]  # Trim padding if added


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
    ):
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
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
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# DLinear model
class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        seq_len = x.size(1)
        padding = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, padding, 1)
        end = x[:, -1:, :].repeat(1, padding, 1)
        x_padded = torch.cat([front, x, end], dim=1)
        x_padded = x_padded.permute(0, 2, 1)
        x_avg = self.avg(x_padded)
        x_avg = x_avg.permute(0, 2, 1)
        return x_avg


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
    segment_size=24,
    mix_rate=0.15,
    scale_factor=0.05,
    variance_threshold=0.03,
    corr_threshold=0.7,
    sampling_rate=0.5,
    epochs=10,
    lr=0.01,
    patience=3,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    aug = Augmentation()
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    path = "./checkpoints/Adaptive-Channel-Preserve"
    if not os.path.exists(path):
        os.makedirs(path)

    for epoch in range(epochs):
        model.train()
        train_loss = []
        epoch_time = time.time()
        for i, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            xy = aug.adaptive_channel_preserve(
                batch_x,
                batch_y[:, -pred_len:, :],
                segment_size=segment_size,
                mix_rate=mix_rate,
                scale_factor=scale_factor,
                variance_threshold=variance_threshold,
                corr_threshold=corr_threshold,
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
        for batch_x, batch_y in val_loader:
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
        for batch_x, batch_y in test_loader:
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
        "aug_types": ["Adaptive-Channel-Preserve"],
        "aug_params": {
            96: {
                "Adaptive-Channel-Preserve": {
                    "segment_size": 24,
                    "mix_rate": 0.15,
                    "scale_factor": 0.05,
                    "variance_threshold": 0.03,
                    "corr_threshold": 0.7,
                    "sampling_rate": 0.5,
                },
            },
            192: {
                "Adaptive-Channel-Preserve": {
                    "segment_size": 24,
                    "mix_rate": 0.15,
                    "scale_factor": 0.05,
                    "variance_threshold": 0.03,
                    "corr_threshold": 0.7,
                    "sampling_rate": 0.5,
                },
            },
            336: {
                "Adaptive-Channel-Preserve": {
                    "segment_size": 36,
                    "mix_rate": 0.1,
                    "scale_factor": 0.05,
                    "variance_threshold": 0.03,
                    "corr_threshold": 0.7,
                    "sampling_rate": 0.7,
                },
            },
            720: {
                "Adaptive-Channel-Preserve": {
                    "segment_size": 48,
                    "mix_rate": 0.1,
                    "scale_factor": 0.05,
                    "variance_threshold": 0.03,
                    "corr_threshold": 0.7,
                    "sampling_rate": 0.7,
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
        "aug_types": ["Adaptive-Channel-Preserve"],
        "aug_params": {
            96: {
                "Adaptive-Channel-Preserve": {
                    "segment_size": 24,
                    "mix_rate": 0.15,
                    "scale_factor": 0.05,
                    "variance_threshold": 0.03,
                    "corr_threshold": 0.7,
                    "sampling_rate": 0.5,
                },
            },
            192: {
                "Adaptive-Channel-Preserve": {
                    "segment_size": 24,
                    "mix_rate": 0.15,
                    "scale_factor": 0.05,
                    "variance_threshold": 0.03,
                    "corr_threshold": 0.7,
                    "sampling_rate": 0.5,
                },
            },
            336: {
                "Adaptive-Channel-Preserve": {
                    "segment_size": 36,
                    "mix_rate": 0.1,
                    "scale_factor": 0.05,
                    "variance_threshold": 0.03,
                    "corr_threshold": 0.7,
                    "sampling_rate": 0.7,
                },
            },
            720: {
                "Adaptive-Channel-Preserve": {
                    "segment_size": 48,
                    "mix_rate": 0.1,
                    "scale_factor": 0.05,
                    "variance_threshold": 0.03,
                    "corr_threshold": 0.7,
                    "sampling_rate": 0.7,
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
        "aug_types": ["Adaptive-Channel-Preserve"],
        "aug_params": {
            96: {
                "Adaptive-Channel-Preserve": {
                    "segment_size": 24,
                    "mix_rate": 0.15,
                    "scale_factor": 0.05,
                    "variance_threshold": 0.03,
                    "corr_threshold": 0.7,
                    "sampling_rate": 0.5,
                },
            },
            192: {
                "Adaptive-Channel-Preserve": {
                    "segment_size": 24,
                    "mix_rate": 0.15,
                    "scale_factor": 0.05,
                    "variance_threshold": 0.03,
                    "corr_threshold": 0.7,
                    "sampling_rate": 1.0,
                },
            },
            336: {
                "Adaptive-Channel-Preserve": {
                    "segment_size": 36,
                    "mix_rate": 0.1,
                    "scale_factor": 0.05,
                    "variance_threshold": 0.03,
                    "corr_threshold": 0.7,
                    "sampling_rate": 1.0,
                },
            },
            720: {
                "Adaptive-Channel-Preserve": {
                    "segment_size": 48,
                    "mix_rate": 0.1,
                    "scale_factor": 0.05,
                    "variance_threshold": 0.03,
                    "corr_threshold": 0.7,
                    "sampling_rate": 0.7,
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
        "aug_types": ["Adaptive-Channel-Preserve"],
        "aug_params": {
            24: {
                "Adaptive-Channel-Preserve": {
                    "segment_size": 6,
                    "mix_rate": 0.1,
                    "scale_factor": 0.03,
                    "variance_threshold": 0.02,
                    "corr_threshold": 0.7,
                    "sampling_rate": 0.5,
                },
            },
            36: {
                "Adaptive-Channel-Preserve": {
                    "segment_size": 6,
                    "mix_rate": 0.1,
                    "scale_factor": 0.03,
                    "variance_threshold": 0.02,
                    "corr_threshold": 0.7,
                    "sampling_rate": 0.5,
                },
            },
            48: {
                "Adaptive-Channel-Preserve": {
                    "segment_size": 6,
                    "mix_rate": 0.1,
                    "scale_factor": 0.03,
                    "variance_threshold": 0.02,
                    "corr_threshold": 0.7,
                    "sampling_rate": 0.5,
                },
            },
            60: {
                "Adaptive-Channel-Preserve": {
                    "segment_size": 6,
                    "mix_rate": 0.1,
                    "scale_factor": 0.03,
                    "variance_threshold": 0.02,
                    "corr_threshold": 0.7,
                    "sampling_rate": 0.5,
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
    num_iterations = 5
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

            # Run experiment for Adaptive-Channel-Preserve
            aug_type = "Adaptive-Channel-Preserve"
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
                    segment_size=params["segment_size"],
                    mix_rate=params["mix_rate"],
                    scale_factor=params["scale_factor"],
                    variance_threshold=params["variance_threshold"],
                    corr_threshold=params["corr_threshold"],
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
                batch_x, batch_y = next(iter(test_loader))
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
        with open(f"results_{dataset_name}_adaptive_channel_preserve.txt", "w") as f:
            for pred_len in config["pred_lens"]:
                for aug_type in config["aug_types"]:
                    metrics = results[(pred_len, aug_type)]
                    f.write(
                        f"{dataset_name}, pred_len={pred_len}, {aug_type}: "
                        f'Val Loss={metrics["val_loss"]:.6f}, MAE={metrics["mae"]:.6f}, MSE={metrics["mse"]:.6f}, '
                        f'RSE={metrics["rse"]:.6f}, MAE Std={metrics["mae_std"]:.6f}, MSE Std={metrics["mse_std"]:.6f}, '
                        f'RSE Std={metrics["rse_std"]:.6f}\n'
                    )
