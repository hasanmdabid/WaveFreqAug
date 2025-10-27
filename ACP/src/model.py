import torch
import torch.nn as nn
import numpy as np


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


# In model.py
class iTransformer(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        d_model=512,
        n_heads=8,
        e_layers=4,
        d_ff=2048,
        dropout=0.1,
    ):
        super(iTransformer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.d_model = d_model

        # Input embedding
        self.input_projection = nn.Linear(enc_in, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_len, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=e_layers
        )

        # Output projection
        self.output_projection = nn.Linear(d_model * seq_len, pred_len * enc_in)

    def forward(self, x):
        batch_size = x.size(0)  # x shape: (batch_size, seq_len, enc_in)

        # Project input to d_model dimension
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        x = (
            x + self.positional_encoding[:, : self.seq_len, :]
        )  # Add positional encoding

        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)

        # Flatten and project to output
        x = x.reshape(batch_size, -1)  # (batch_size, seq_len * d_model)
        x = self.output_projection(x)  # (batch_size, pred_len * enc_in)
        x = x.view(
            batch_size, self.pred_len, self.enc_in
        )  # (batch_size, pred_len, enc_in)

        return x


# Metrics
def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(
        np.sum((true - true.mean()) ** 2)
    )


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)
