# =========================================================================================
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
import gc
from baseline.utils.aug_methods import Augmentation


def _device_type(device):
    return torch.device(device).type


def _empty_cache(device_type):
    if device_type == "cuda":
        torch.cuda.empty_cache()
    elif device_type == "mps":
        torch.mps.empty_cache()


# Metrics
def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(
        np.sum((true - true.mean()) ** 2)
    )


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


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
    dominant_k=4,
    epochs=30,
    lr=0.01,
    patience=12,
    checkpoint_dir=None,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    device_type = _device_type(device)
    use_amp = device_type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)
    aug = Augmentation()
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    path = checkpoint_dir if checkpoint_dir is not None else f"./checkpoints/{aug_type}"
    if not os.path.exists(path):
        os.makedirs(path)
    # Clear any checkpoint left over from a previous iteration of this same combo
    # so a run that diverges before ever saving one can't silently be scored
    # against a stale, unrelated checkpoint.
    stale_ckpt = os.path.join(path, "checkpoint.pth")
    if os.path.exists(stale_ckpt):
        os.remove(stale_ckpt)

    for epoch in range(epochs):
        model.train()
        train_loss = []
        epoch_time = time.time()
        for i, (batch_x, batch_y, aug_data) in enumerate(train_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            aug_data = aug_data.float().to(device) if aug_data is not None else None

            # Initialize variables for cleanup
            xy = None
            batch_x2 = None
            batch_y2 = None
            loss_aug = None

            optimizer.zero_grad()
            with autocast(device_type=device_type, enabled=use_amp):
                if aug_type == "None":
                    outputs = model(batch_x)
                    loss = criterion(
                        outputs[:, -pred_len:, :], batch_y[:, -pred_len:, :]
                    )
                else:
                    # Original batch
                    outputs = model(batch_x)
                    loss = criterion(
                        outputs[:, -pred_len:, :], batch_y[:, -pred_len:, :]
                    )
                    loss = loss / 2

                    # Augmented batch
                    if aug_type == "Freq-Mask":
                        xy = aug.freq_mask(
                            batch_x.cpu(),
                            batch_y[:, -pred_len:, :].cpu(),
                            rate=aug_rate,
                            dim=1,
                        )
                        batch_x2, batch_y2 = (
                            xy[:, :seq_len, :].to(device),
                            xy[:, seq_len : seq_len + label_len + pred_len, :].to(
                                device
                            ),
                        )
                    elif aug_type == "Freq-Mix":
                        xy = aug.freq_mix(
                            batch_x.cpu(),
                            batch_y[:, -pred_len:, :].cpu(),
                            rate=aug_rate,
                            dim=1,
                        )
                        batch_x2, batch_y2 = (
                            xy[:, :seq_len, :].to(device),
                            xy[:, seq_len : seq_len + label_len + pred_len, :].to(
                                device
                            ),
                        )
                    elif aug_type == "Wave-Mask":
                        xy = aug.wave_mask(
                            batch_x.cpu(),
                            batch_y[:, -pred_len:, :].cpu(),
                            rates=rates,
                            wavelet=wavelet,
                            level=level,
                            dim=1,
                        )
                        sampling_steps = int(batch_x.shape[0] * sampling_rate)
                        indices = torch.randperm(batch_x.shape[0])[:sampling_steps]
                        batch_x2, batch_y2 = (
                            xy[indices, :seq_len, :].to(device),
                            xy[indices, seq_len : seq_len + label_len + pred_len, :].to(
                                device
                            ),
                        )
                    elif aug_type == "Wave-Mix":
                        xy = aug.wave_mix(
                            batch_x.cpu(),
                            batch_y[:, -pred_len:, :].cpu(),
                            rates=rates,
                            wavelet=wavelet,
                            level=level,
                            dim=1,
                        )
                        sampling_steps = int(batch_x.shape[0] * sampling_rate)
                        indices = torch.randperm(batch_x.shape[0])[:sampling_steps]
                        batch_x2, batch_y2 = (
                            xy[indices, :seq_len, :].to(device),
                            xy[indices, seq_len : seq_len + label_len + pred_len, :].to(
                                device
                            ),
                        )
                    elif aug_type == "Dominant-Shuffle":
                        xy = aug.dominant_shuffle(
                            batch_x.cpu(),
                            batch_y[:, -pred_len:, :].cpu(),
                            k=dominant_k,
                            dim=1,
                        )
                        batch_x2, batch_y2 = (
                            xy[:, :seq_len, :].to(device),
                            xy[:, seq_len : seq_len + label_len + pred_len, :].to(
                                device
                            ),
                        )
                    elif aug_type == "StAug":
                        weighted_xy = aug.emd_aug(aug_data)  # type: ignore
                        batch_x2, batch_y2 = aug.mix_aug(
                            weighted_xy[:, :seq_len, :],
                            weighted_xy[:, seq_len : seq_len + label_len + pred_len, :],
                            lambd=aug_rate,
                        )
                    outputs = model(batch_x2)
                    loss_aug = criterion(
                        outputs[:, -pred_len:, :], batch_y2[:, -pred_len:, :] # type: ignore
                    )
                    loss_aug = loss_aug / 2
                    loss = loss + loss_aug

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss.append(loss.item())

            # Clean up variables to prevent memory leaks
            del batch_x, batch_y, outputs, loss
            if aug_type != "None":
                if xy is not None:
                    del xy
                if batch_x2 is not None:
                    del batch_x2
                if batch_y2 is not None:
                    del batch_y2
                if loss_aug is not None:
                    del loss_aug
            _empty_cache(device_type)
            gc.collect()

        train_loss = np.average(train_loss)
        val_loss = validate(model, val_loader, device, criterion, pred_len)
        print(
            f"Epoch: {epoch + 1}, Time: {time.time() - epoch_time:.2f}s | Train Loss: {train_loss:.7f} Val Loss: {val_loss:.7f}"
        )

        if not np.isfinite(val_loss):
            # Diverged. NaN/Inf breaks EarlyStopping's comparisons (any
            # comparison against nan is False, so it would otherwise never
            # patience-stop and would keep overwriting the checkpoint with
            # diverged weights every epoch) — bail out now instead of burning
            # the rest of the epoch budget on a run that's already dead.
            print(f"Epoch: {epoch + 1}: val_loss is non-finite ({val_loss}) — aborting this run.")
            break

        early_stopping(val_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        adjust_learning_rate(optimizer, epoch + 1, lr)

    return early_stopping.get_val_loss_min()


def validate(model, val_loader, device, criterion, pred_len):
    model.eval()
    total_loss = []
    device_type = _device_type(device)
    use_amp = device_type == "cuda"
    with torch.no_grad():
        for batch_x, batch_y, _ in val_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            with autocast(device_type=device_type, enabled=use_amp):
                outputs = model(batch_x)
                loss = criterion(outputs[:, -pred_len:, :], batch_y[:, -pred_len:, :])
            total_loss.append(loss.item())
            del batch_x, batch_y, outputs, loss
            _empty_cache(device_type)
            gc.collect()
    return np.average(total_loss)


def test(model, test_loader, device, scaler, pred_len):
    model.eval()
    preds, trues = [], []
    device_type = _device_type(device)
    use_amp = device_type == "cuda"
    with torch.no_grad():
        for batch_x, batch_y, _ in test_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            with autocast(device_type=device_type, enabled=use_amp):
                outputs = model(batch_x)
            outputs = outputs[:, -pred_len:, :].cpu().numpy()
            batch_y = batch_y[:, -pred_len:, :].cpu().numpy()
            preds.append(outputs)
            trues.append(batch_y)
            del batch_x, batch_y, outputs
            _empty_cache(device_type)
            gc.collect()

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    mae = MAE(preds, trues)
    mse = MSE(preds, trues)
    rse = RSE(preds, trues)
    return mae, mse, rse
