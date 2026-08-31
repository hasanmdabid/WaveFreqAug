# =========================================================================================
# This script is written and organized by Md Abid Hasan towards the project WaveFreqAug.
# The DLinear and iTransformer models are adapted from the following sources:
# =========================================================================================

import math
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

#------------------------------------------------------------Model 1---------------------------------------------------------------------------------------- 

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


# SCINet model — faithful port of the official recursive-tree architecture
# (Liu et al., NeurIPS 2022; https://github.com/cure-lab/SCINet), matching
# https://github.com/zuojie2024/dominant-shuffle/blob/main/models/SCINet.py
# (the version used to produce the Dominant-Shuffle baseline this project
# compares against). The official reference class takes an argparse
# `Namespace` and immediately overwrites its own kwargs with `args.*` fields;
# that indirection is flattened here into plain kwargs to match this
# project's calling convention — no computation is changed, only how config
# values reach __init__.
class Splitting(nn.Module):
    def even(self, x):
        return x[:, ::2, :]

    def odd(self, x):
        return x[:, 1::2, :]

    def forward(self, x):
        return self.even(x), self.odd(x)


class Interactor(nn.Module):
    def __init__(self, in_planes, splitting=True, kernel=5, dropout=0.5,
                 groups=1, hidden_size=1, INN=True):
        super(Interactor, self).__init__()
        self.modified = INN
        self.kernel_size = kernel
        self.dilation = 1
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.groups = groups
        if self.kernel_size % 2 == 0:
            pad_l = self.dilation * (self.kernel_size - 2) // 2 + 1
            pad_r = self.dilation * self.kernel_size // 2 + 1
        else:
            pad_l = self.dilation * (self.kernel_size - 1) // 2 + 1
            pad_r = self.dilation * (self.kernel_size - 1) // 2 + 1
        self.splitting = splitting
        self.split = Splitting()

        size_hidden = self.hidden_size

        def _branch():
            return nn.Sequential(
                nn.ReplicationPad1d((pad_l, pad_r)),
                nn.Conv1d(in_planes, int(in_planes * size_hidden), kernel_size=self.kernel_size,
                          dilation=self.dilation, stride=1, groups=self.groups),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(self.dropout),
                nn.Conv1d(int(in_planes * size_hidden), in_planes, kernel_size=3, stride=1, groups=self.groups),
                nn.Tanh(),
            )

        self.phi = _branch()
        self.psi = _branch()
        self.P = _branch()
        self.U = _branch()

    def forward(self, x):
        if self.splitting:
            x_even, x_odd = self.split(x)
        else:
            x_even, x_odd = x

        x_even = x_even.permute(0, 2, 1)
        x_odd = x_odd.permute(0, 2, 1)

        if self.modified:
            d = x_odd.mul(torch.exp(self.phi(x_even)))
            c = x_even.mul(torch.exp(self.psi(x_odd)))
            x_even_update = c + self.U(d)
            x_odd_update = d - self.P(c)
            return x_even_update, x_odd_update
        else:
            d = x_odd - self.P(x_even)
            c = x_even + self.U(d)
            return c, d


class InteractorLevel(nn.Module):
    def __init__(self, in_planes, kernel, dropout, groups, hidden_size, INN):
        super(InteractorLevel, self).__init__()
        self.level = Interactor(in_planes=in_planes, splitting=True, kernel=kernel,
                                 dropout=dropout, groups=groups, hidden_size=hidden_size, INN=INN)

    def forward(self, x):
        return self.level(x)


class LevelSCINet(nn.Module):
    def __init__(self, in_planes, kernel_size, dropout, groups, hidden_size, INN):
        super(LevelSCINet, self).__init__()
        self.interact = InteractorLevel(in_planes=in_planes, kernel=kernel_size, dropout=dropout,
                                         groups=groups, hidden_size=hidden_size, INN=INN)

    def forward(self, x):
        x_even_update, x_odd_update = self.interact(x)
        return x_even_update.permute(0, 2, 1), x_odd_update.permute(0, 2, 1)  # both: B, T, D


class SCINet_Tree(nn.Module):
    def __init__(self, in_planes, current_level, kernel_size, dropout, groups, hidden_size, INN):
        super(SCINet_Tree, self).__init__()
        self.current_level = current_level
        self.workingblock = LevelSCINet(in_planes=in_planes, kernel_size=kernel_size, dropout=dropout,
                                         groups=groups, hidden_size=hidden_size, INN=INN)
        if current_level != 0:
            self.SCINet_Tree_odd = SCINet_Tree(in_planes, current_level - 1, kernel_size, dropout, groups, hidden_size, INN)
            self.SCINet_Tree_even = SCINet_Tree(in_planes, current_level - 1, kernel_size, dropout, groups, hidden_size, INN)

    def zip_up_the_pants(self, even, odd):
        even = even.permute(1, 0, 2)
        odd = odd.permute(1, 0, 2)  # L, B, D
        even_len, odd_len = even.shape[0], odd.shape[0]
        mlen = min(odd_len, even_len)
        zipped = []
        for i in range(mlen):
            zipped.append(even[i].unsqueeze(0))
            zipped.append(odd[i].unsqueeze(0))
        if odd_len < even_len:
            zipped.append(even[-1].unsqueeze(0))
        return torch.cat(zipped, 0).permute(1, 0, 2)  # B, L, D

    def forward(self, x):
        x_even_update, x_odd_update = self.workingblock(x)
        if self.current_level == 0:
            return self.zip_up_the_pants(x_even_update, x_odd_update)
        return self.zip_up_the_pants(
            self.SCINet_Tree_even(x_even_update), self.SCINet_Tree_odd(x_odd_update)
        )


class EncoderTree(nn.Module):
    def __init__(self, in_planes, num_levels, kernel_size, dropout, groups, hidden_size, INN):
        super(EncoderTree, self).__init__()
        self.SCINet_Tree = SCINet_Tree(in_planes, num_levels - 1, kernel_size, dropout, groups, hidden_size, INN)

    def forward(self, x):
        return self.SCINet_Tree(x)


#------------------------------------------------------------Model 2----------------------------------------------------------------------------------------
class SCINet(nn.Module):
    def __init__(
        self, input_len, output_len, input_dim, hid_size=1, num_stacks=1,
        num_levels=3, kernel_size=5, dropout=0.5, num_decoder_layer=1,
        concat_len=0, groups=1, single_step_output_One=0,
        positionalE=False, modified=True, RIN=False,
    ):
        super(SCINet, self).__init__()
        assert num_stacks in (1, 2), "SCINet supports 1 or 2 stacks only"
        self.input_dim = input_dim

        # Every recursive odd/even split in SCINet_Tree must land on an even
        # length all the way down to the leaves, i.e. input_len must be an
        # exact multiple of 2**num_levels (the official implementation
        # asserts this rather than handling shorter inputs). Left-pad by
        # repeating the first real time step so the true most-recent step —
        # used below for the last-value residual — is unaffected. ILI's
        # seq_len=36 needs this (36 is not a multiple of 2**3=8); ETTh1/ETTh2/
        # weather's seq_len=336 already is and this is a no-op for them.
        pad_unit = 2 ** num_levels
        self.padded_input_len = ((input_len + pad_unit - 1) // pad_unit) * pad_unit
        self.pad_amount = self.padded_input_len - input_len

        self.input_len = self.padded_input_len
        self.output_len = output_len
        self.hidden_size = hid_size
        self.num_levels = num_levels
        self.groups = groups
        self.modified = modified
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.num_decoder_layer = num_decoder_layer
        self.concat_len = concat_len
        self.pe = positionalE
        self.RIN = RIN
        self.single_step_output_One = single_step_output_One
        self.stacks = num_stacks

        self.blocks1 = EncoderTree(
            in_planes=self.input_dim, num_levels=self.num_levels, kernel_size=self.kernel_size,
            dropout=self.dropout, groups=self.groups, hidden_size=self.hidden_size, INN=modified,
        )
        if self.stacks == 2:
            self.blocks2 = EncoderTree(
                in_planes=self.input_dim, num_levels=self.num_levels, kernel_size=self.kernel_size,
                dropout=self.dropout, groups=self.groups, hidden_size=self.hidden_size, INN=modified,
            )

        self.projection1 = nn.Conv1d(self.input_len, self.output_len, kernel_size=1, stride=1, bias=False)
        self.div_projection = nn.ModuleList()
        self.overlap_len = self.input_len // 4
        self.div_len = self.input_len // 6

        if self.num_decoder_layer > 1:
            self.projection1 = nn.Linear(self.input_len, self.output_len)
            for _ in range(self.num_decoder_layer - 1):
                div_projection = nn.ModuleList()
                for i in range(6):
                    lens = min(i * self.div_len + self.overlap_len, self.input_len) - i * self.div_len
                    div_projection.append(nn.Linear(lens, self.div_len))
                self.div_projection.append(div_projection)

        if self.stacks == 2:
            proj2_in = (self.concat_len if self.concat_len else self.input_len) + self.output_len
            proj2_out = 1 if self.single_step_output_One else self.output_len
            self.projection2 = nn.Conv1d(proj2_in, proj2_out, kernel_size=1, bias=False)

        # Positional-encoding buffer (only used when positionalE=True).
        self.pe_hidden_size = self.input_dim + (self.input_dim % 2)
        num_timescales = self.pe_hidden_size // 2
        max_timescale, min_timescale = 10000.0, 1.0
        log_timescale_increment = math.log(max_timescale / min_timescale) / max(num_timescales - 1, 1)
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) * -log_timescale_increment
        )
        self.register_buffer("inv_timescales", inv_timescales)

        if self.RIN:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, input_dim))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, input_dim))

    def get_position_encoding(self, x):
        max_length = x.size(1)
        position = torch.arange(max_length, dtype=torch.float32, device=x.device)
        scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
        signal = F.pad(signal, (0, 0, 0, self.pe_hidden_size % 2))
        return signal.view(1, max_length, self.pe_hidden_size)

    def forward(self, x):
        # x: (batch, raw seq_len, input_dim)
        if self.pad_amount > 0:
            front = x[:, :1, :].repeat(1, self.pad_amount, 1)
            x = torch.cat([front, x], dim=1)

        last_value = x[:, -1:, :].detach()
        x = x - last_value
        if self.pe:
            pe = self.get_position_encoding(x)
            x = x + (pe[:, :, :-1] if pe.shape[2] > x.shape[2] else pe)

        if self.RIN:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev
            x = x * self.affine_weight + self.affine_bias

        res1 = x
        x = self.blocks1(x)
        x = x + res1
        if self.num_decoder_layer == 1:
            x = self.projection1(x)
        else:
            x = x.permute(0, 2, 1)
            for div_projection in self.div_projection:
                output = torch.zeros_like(x)
                for i, div_layer in enumerate(div_projection):
                    div_x = x[:, :, i * self.div_len:min(i * self.div_len + self.overlap_len, self.input_len)]
                    output[:, :, i * self.div_len:(i + 1) * self.div_len] = div_layer(div_x)
                x = output
            x = self.projection1(x)
            x = x.permute(0, 2, 1)

        if self.stacks == 1:
            if self.RIN:
                x = x - self.affine_bias
                x = x / (self.affine_weight + 1e-10)
                x = x * stdev
                x = x + means
            return x + last_value

        # stacks == 2. Only the final-stack output is returned — the official
        # reference also returns the mid-stack output as a second value, but
        # every model in this codebase shares a single-tensor forward()
        # contract, and this project only ever uses num_stacks=1.
        if self.concat_len:
            x2 = torch.cat((res1[:, -self.concat_len:, :], x), dim=1)
        else:
            x2 = torch.cat((res1, x), dim=1)
        res2 = x2
        x2 = self.blocks2(x2)
        x2 = x2 + res2
        x2 = self.projection2(x2)
        if self.RIN:
            x2 = x2 - self.affine_bias
            x2 = x2 / (self.affine_weight + 1e-10)
            x2 = x2 * stdev
            x2 = x2 + means
        return x2 + last_value

#------------------------------------------------------------Model 3----------------------------------------------------------------------------------------
# iTransformer (adapted from https://github.com/thuml/iTransformer)
# Inverts the attention axis: each variate's full history is embedded as a single
# token, so self-attention mixes across variates rather than across time steps.
class iTransformer(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        d_model=512,
        n_heads=8,
        e_layers=2,
        d_ff=512,
        dropout=0.1,
    ):
        super(iTransformer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in

        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,  # pre-LN: standard/official iTransformer practice, far more
                              # stable than PyTorch's post-LN default at this lr.
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.norm = nn.LayerNorm(d_model)
        self.projector = nn.Linear(d_model, pred_len)

    def forward(self, x):
        # x: (batch_size, seq_len, enc_in)
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        tokens = x.permute(0, 2, 1)  # (batch_size, enc_in, seq_len)
        tokens = self.dropout(self.value_embedding(tokens))  # (batch_size, enc_in, d_model)
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        dec_out = self.projector(tokens)  # (batch_size, enc_in, pred_len)
        dec_out = dec_out.permute(0, 2, 1)  # (batch_size, pred_len, enc_in)

        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out


#------------------------------------------------------------Model 4----------------------------------------------------------------------------------------
# FEDformer (adapted from https://github.com/MAZiqing/FEDformer)
# Encoder-decoder with moving-average trend/seasonal decomposition, where the
# self- and cross-attention sublayers (FEB-f / FEA-f) operate on a truncated set
# of low-frequency Fourier modes instead of raw dot-product attention.
class FourierBlock(nn.Module):
    def __init__(self, d_model, n_heads, modes=32):
        super(FourierBlock, self).__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.modes = modes
        scale = 1 / (self.d_head * self.d_head)
        self.weights_real = nn.Parameter(scale * torch.randn(n_heads, self.d_head, self.d_head, modes))
        self.weights_imag = nn.Parameter(scale * torch.randn(n_heads, self.d_head, self.d_head, modes))

    def forward(self, x):
        # x: (B, L, D)
        B, L, D = x.shape
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            xh = x.view(B, L, self.n_heads, self.d_head).permute(0, 2, 3, 1)  # B,H,E,L
            x_ft = torch.fft.rfft(xh, dim=-1)
            modes = min(self.modes, x_ft.shape[-1])
            weight = torch.complex(self.weights_real[..., :modes], self.weights_imag[..., :modes])
            out_ft = torch.zeros(B, self.n_heads, self.d_head, x_ft.shape[-1], dtype=torch.cfloat, device=x.device)
            out_ft[..., :modes] = torch.einsum('bhem,heom->bhom', x_ft[..., :modes], weight)
            out = torch.fft.irfft(out_ft, n=L, dim=-1)
        return out.permute(0, 3, 1, 2).reshape(B, L, D)


class FourierCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, modes=32):
        super(FourierCrossAttention, self).__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.modes = modes

    def forward(self, q, k, v):
        # q: (B, Lq, D)   k, v: (B, Lk, D)
        B, Lq, D = q.shape
        Lk = k.shape[1]
        H, E = self.n_heads, self.d_head
        with torch.autocast(device_type=q.device.type, enabled=False):
            q, k, v = q.float(), k.float(), v.float()
            qh = q.view(B, Lq, H, E).permute(0, 2, 3, 1)
            kh = k.view(B, Lk, H, E).permute(0, 2, 3, 1)
            vh = v.view(B, Lk, H, E).permute(0, 2, 3, 1)

            q_ft = torch.fft.rfft(qh, dim=-1)
            k_ft = torch.fft.rfft(kh, dim=-1)
            v_ft = torch.fft.rfft(vh, dim=-1)
            modes = min(self.modes, q_ft.shape[-1], k_ft.shape[-1])
            q_m, k_m, v_m = q_ft[..., :modes], k_ft[..., :modes], v_ft[..., :modes]

            score = torch.einsum('bhem,bhen->bhmn', q_m, k_m.conj())
            att = torch.softmax(score.abs(), dim=-1).to(torch.cfloat)
            out_m = torch.einsum('bhmn,bhen->bhem', att, v_m)

            out_ft = torch.zeros(B, H, E, q_ft.shape[-1], dtype=torch.cfloat, device=q.device)
            out_ft[..., :modes] = out_m
            out = torch.fft.irfft(out_ft, n=Lq, dim=-1)
        return out.permute(0, 3, 1, 2).reshape(B, Lq, D)


class FEDEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, moving_avg=25, modes=32, dropout=0.1):
        super(FEDEncoderLayer, self).__init__()
        self.attn = FourierBlock(d_model, n_heads, modes)
        self.q_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.decomp1 = SeriesDecomp(moving_avg)
        self.decomp2 = SeriesDecomp(moving_avg)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        new_x = self.out_proj(self.attn(self.q_proj(x)))
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        y = self.dropout(F.gelu(self.ff1(x)))
        y = self.dropout(self.ff2(y))
        res, _ = self.decomp2(x + y)
        return res


class FEDDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, c_out, d_ff, moving_avg=25, modes=32, dropout=0.1):
        super(FEDDecoderLayer, self).__init__()
        self.self_attn = FourierBlock(d_model, n_heads, modes)
        self.self_q_proj = nn.Linear(d_model, d_model)
        self.self_out_proj = nn.Linear(d_model, d_model)

        self.cross_attn = FourierCrossAttention(d_model, n_heads, modes)
        self.cross_q_proj = nn.Linear(d_model, d_model)
        self.cross_k_proj = nn.Linear(d_model, d_model)
        self.cross_v_proj = nn.Linear(d_model, d_model)
        self.cross_out_proj = nn.Linear(d_model, d_model)

        self.decomp1 = SeriesDecomp(moving_avg)
        self.decomp2 = SeriesDecomp(moving_avg)
        self.decomp3 = SeriesDecomp(moving_avg)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.trend_proj = nn.Linear(d_model, c_out)

    def forward(self, x, cross, trend_accum):
        new_x = self.self_out_proj(self.self_attn(self.self_q_proj(x)))
        x = x + self.dropout(new_x)
        x, trend1 = self.decomp1(x)

        new_x = self.cross_out_proj(self.cross_attn(
            self.cross_q_proj(x), self.cross_k_proj(cross), self.cross_v_proj(cross)
        ))
        x = x + self.dropout(new_x)
        x, trend2 = self.decomp2(x)

        y = self.dropout(F.gelu(self.ff1(x)))
        y = self.dropout(self.ff2(y))
        x, trend3 = self.decomp3(x + y)

        trend_accum = trend_accum + self.trend_proj(trend1 + trend2 + trend3)
        return x, trend_accum


class FEDformer(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, label_len=None, d_model=64,
                 n_heads=8, e_layers=2, d_layers=1, d_ff=64, moving_avg=25,
                 modes=32, dropout=0.1):
        super(FEDformer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = max(label_len if label_len is not None else min(48, seq_len // 2), 1)

        self.decomp = SeriesDecomp(moving_avg)
        self.enc_embedding = nn.Linear(enc_in, d_model)
        self.dec_embedding = nn.Linear(enc_in, d_model)

        self.encoder_layers = nn.ModuleList([
            FEDEncoderLayer(d_model, n_heads, d_ff, moving_avg, modes, dropout)
            for _ in range(e_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            FEDDecoderLayer(d_model, n_heads, enc_in, d_ff, moving_avg, modes, dropout)
            for _ in range(d_layers)
        ])
        self.seasonal_projection = nn.Linear(d_model, enc_in)

    def forward(self, x):
        # x: (batch_size, seq_len, enc_in) -> returns (batch_size, label_len+pred_len, enc_in)
        B, _, C = x.shape
        seasonal_init, trend_init = self.decomp(x)

        mean = torch.mean(x, dim=1, keepdim=True).repeat(1, self.pred_len, 1)
        zeros = torch.zeros(B, self.pred_len, C, device=x.device, dtype=x.dtype)
        dec_seasonal = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        dec_trend = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)

        enc_out = self.enc_embedding(x)
        for layer in self.encoder_layers:
            enc_out = layer(enc_out)

        dec_out = self.dec_embedding(dec_seasonal)
        trend_accum = dec_trend
        for layer in self.decoder_layers:
            dec_out, trend_accum = layer(dec_out, enc_out, trend_accum)

        seasonal_out = self.seasonal_projection(dec_out)
        return seasonal_out + trend_accum
