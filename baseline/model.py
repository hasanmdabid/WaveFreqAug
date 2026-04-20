# =========================================================================================
# This script is written and organized by Md Abid Hasan towards the project WaveFreqAug.
# The DLinear and iTransformer models are adapted from the following sources:
# =========================================================================================

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

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

# SCINet model (adapted from https://github.com/cure-lab/SCINet)
class SCIBlock(nn.Module):
    def __init__(self, input_dim, hid_size, kernel_size=5, dropout=0.2):
        super(SCIBlock, self).__init__()
        self.input_dim = input_dim
        self.hid_size = hid_size
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.conv1 = nn.Conv1d(input_dim, hid_size, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(input_dim, hid_size, kernel_size, padding=kernel_size//2)
        self.fc = nn.Linear(hid_size, hid_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hid_size)

    def forward(self, x):
        # x: [batch_size, input_dim, seq_len]
        x_even = x[:, :, ::2]  # Downsample: even indices
        x_odd = x[:, :, 1::2]  # Downsample: odd indices
        x_even = self.conv1(x_even)
        x_odd = self.conv2(x_odd)
        x_even = self.norm(x_even.permute(0, 2, 1)).permute(0, 2, 1)
        x_odd = self.norm(x_odd.permute(0, 2, 1)).permute(0, 2, 1)
        x_even = torch.tanh(x_even) * torch.sigmoid(x_odd)
        x_odd = torch.tanh(x_odd) * torch.sigmoid(x_even)
        x_even = self.dropout(self.fc(x_even.permute(0, 2, 1)).permute(0, 2, 1)) # type: ignore
        x_odd = self.dropout(self.fc(x_odd.permute(0, 2, 1)).permute(0, 2, 1)) # type: ignore
        return x_even, x_odd

class SCINet(nn.Module):
    def __init__(self, input_len, output_len, input_dim, hid_size=1, num_stacks=1, num_levels=3, kernel_size=5, dropout=0.2):
        super(SCINet, self).__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.input_dim = input_dim
        self.hid_size = hid_size
        self.num_stacks = num_stacks
        self.num_levels = num_levels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.blocks = nn.ModuleList()
        for stack in range(num_stacks):
            for level in range(num_levels):
                self.blocks.append(SCIBlock(input_dim if level == 0 else hid_size, hid_size, kernel_size, dropout))
        self.fc = nn.Linear(input_len * hid_size, output_len * input_dim)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        x = x.permute(0, 2, 1)  # [batch_size, input_dim, seq_len]
        for block in self.blocks:
            x_even, x_odd = block(x)
            x = torch.cat([x_even, x_odd], dim=2)  # Concatenate even and odd features
        x = x.permute(0, 2, 1)  # [batch_size, seq_len, hid_size]
        x = x.reshape(x.size(0), -1)  # Flatten: [batch_size, seq_len * hid_size]
        x = self.fc(x)  # [batch_size, output_len * input_dim]
        x = x.reshape(x.size(0), self.output_len, self.input_dim)  # [batch_size, output_len, input_dim]
        return x