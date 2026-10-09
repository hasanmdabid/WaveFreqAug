# This script is written and organized by Md Abid Hasan towards the project WaveFreqAug.
# This is script for training and evaluating time series forecasting models with
# various data augmentation techniques.
# =========================================================================================

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast # type: ignore
import numpy as np
import os
import time
from main.utils.aug_method import Augmentation
from main.utils.dataset_parameter import dataset_configs


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(
        np.sum((true - true.mean()) ** 2)
    )

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    return np.mean((pred - true) ** 2)


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
        os.makedirs(path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(path, "checkpoint.pth"))
        self.val_loss_min = val_loss

    def get_val_loss_min(self):
        return self.val_loss_min


def adjust_learning_rate(optimizer, epoch, lr):
    lr_adjust = {epoch: lr * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust:
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print(f"Updating learning rate to {lr}")


def train(
    model,
    train_loader,
    val_loader,
    device,
    aug_type,
    seq_len,
    label_len,
    pred_len,
    dataset_name,
    mask_rate,
    wavelet,
    level,
    sampling_rate,
    lambd="U-Shape",
    window=12,
    epochs=30,
    lr=0.01,
    patience=12,
    checkpoint_dir="./checkpoints/Wave-Freq",
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss()
    scaler = GradScaler("cuda")
    aug = Augmentation()
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Clear any checkpoint left over from a previous iteration of this same combo
    # so a run that diverges before ever saving one can't silently be scored
    # against a stale, unrelated checkpoint.
    stale_ckpt = os.path.join(checkpoint_dir, "checkpoint.pth")
    if os.path.exists(stale_ckpt):
        os.remove(stale_ckpt)

    diverged = False
    for epoch in range(epochs):
        model.train()
        train_loss = []
        epoch_time = time.time()

        for batch_x, batch_y in train_loader:
            batch_x_cpu = batch_x.float()
            batch_y_cpu = batch_y.float()
            batch_x = batch_x_cpu.to(device, non_blocking=True)
            batch_y = batch_y_cpu.to(device, non_blocking=True)

            # Run CPU augmentation while the async transfer is completing.
            if aug_type == "Wave-Freq":
                xy = aug.wave_freq_aug(
                    batch_x_cpu,
                    batch_y_cpu[:, -pred_len:, :],
                    mask_rate=mask_rate,
                    wavelet=wavelet,
                    level=level,
                    lambd=lambd,
                    dim=1,
                    window=window,
                    top_k_ratio=sampling_rate,
                )
                sampling_steps = int(batch_x_cpu.shape[0] * sampling_rate)
                indices = torch.randperm(batch_x_cpu.shape[0])[:sampling_steps]
                batch_x2 = xy[indices, :seq_len, :].to(device, non_blocking=True)
                batch_y2 = xy[indices, seq_len : seq_len + label_len + pred_len, :].to(
                    device, non_blocking=True
                )

            optimizer.zero_grad()
            with autocast(device_type="cuda"):
                outputs = model(batch_x)
                loss = criterion(outputs[:, -pred_len:, :], batch_y[:, -pred_len:, :])
                if aug_type == "Wave-Freq":
                    loss = loss / 2
                    outputs2 = model(batch_x2)
                    loss_aug = criterion(
                        outputs2[:, -pred_len:, :], batch_y2[:, -pred_len:, :]
                    )
                    loss = loss + loss_aug / 2

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss.append(loss.item())

        train_loss = np.average(train_loss)
        val_loss = validate(model, val_loader, device, criterion, pred_len)
        print(
            f"Epoch: {epoch + 1}, Time: {time.time() - epoch_time:.2f}s | "
            f"Train Loss: {train_loss:.7f} Val Loss: {val_loss:.7f}"
        )

        if not np.isfinite(val_loss):

            print(f"Epoch: {epoch + 1}: val_loss is non-finite ({val_loss}) — aborting this run.")
            diverged = True
            break

        early_stopping(val_loss, model, checkpoint_dir)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        adjust_learning_rate(optimizer, epoch + 1, lr)

    if not diverged:
        print(f"Saving final checkpoint to {os.path.join(checkpoint_dir, 'checkpoint.pth')}")
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "checkpoint.pth"))
    return early_stopping.get_val_loss_min()


def validate(model, val_loader, device, criterion, pred_len):
    model.eval()
    total_loss = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.float().to(device, non_blocking=True)
            batch_y = batch_y.float().to(device, non_blocking=True)
            with autocast(device_type="cuda"):
                outputs = model(batch_x)
                loss = criterion(outputs[:, -pred_len:, :], batch_y[:, -pred_len:, :])
            total_loss.append(loss.item())
    return np.average(total_loss)


def test(model, test_loader, device, scaler, pred_len):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.float().to(device, non_blocking=True)
            batch_y = batch_y.float().to(device, non_blocking=True)
            with autocast(device_type="cuda"):
                outputs = model(batch_x)
            preds.append(outputs[:, -pred_len:, :].cpu().numpy())
            trues.append(batch_y[:, -pred_len:, :].cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    return MAE(preds, trues), MSE(preds, trues), RSE(preds, trues)
