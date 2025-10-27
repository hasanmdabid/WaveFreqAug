import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast # type: ignore 
import numpy as np
import os
import time
import gc
from aug_method import Augmentation
from dataset_parameter import dataset_configs


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
        os.makedirs(path, exist_ok=True)
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
    dataset_name,
    mask_rate,  # Added
    wavelet,    # Added
    level,      # Added
    sampling_rate,  # Added
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

    for epoch in range(epochs):
        model.train()
        train_loss = []
        epoch_time = time.time()
        for i, (batch_x, batch_y) in enumerate(train_loader):
            print(
                f"Batch {i} - GPU memory allocated: {torch.cuda.memory_allocated(device)/1e9:.2f} GB"
            )
            batch_x = batch_x.float().to(device)  # Shape: (batch_size, seq_len, enc_in)
            batch_y = batch_y.float().to(device)  # Shape: (batch_size, label_len + pred_len, enc_in)

            optimizer.zero_grad()
            with autocast(device_type="cuda"):
                outputs = model(batch_x)
                loss = criterion(outputs[:, -pred_len:, :], batch_y[:, -pred_len:, :])

                if aug_type == "Wave-Freq":
                    loss = loss / 2
                    xy = aug.wave_freq_aug(
                        batch_x.cpu(),
                        batch_y[:, -pred_len:, :].cpu(),
                        mask_rate=mask_rate,  # Use grid search parameter
                        wavelet=wavelet,      # Use grid search parameter
                        level=level,          # Use grid search parameter
                        lambd=None,
                        dim=1,
                        window=12,
                        top_k_ratio=sampling_rate,  # Use grid search parameter
                    )
                    sampling_steps = int(batch_x.shape[0] * sampling_rate)
                    indices = torch.randperm(batch_x.shape[0])[:sampling_steps]
                    batch_x2, batch_y2 = (
                        xy[indices, :seq_len, :].to(device),
                        xy[indices, seq_len : seq_len + label_len + pred_len, :].to(device),
                    )
                    outputs = model(batch_x2)
                    loss_aug = criterion(
                        outputs[:, -pred_len:, :], batch_y2[:, -pred_len:, :]
                    )
                    loss = loss + loss_aug / 2

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss.append(loss.item())

            del batch_x, batch_y, outputs, loss
            if aug_type == "Wave-Freq":
                del xy, batch_x2, batch_y2, loss_aug
            torch.cuda.empty_cache()
            gc.collect()

        train_loss = np.average(train_loss)
        val_loss = validate(model, val_loader, device, criterion, pred_len)
        print(
            f"Epoch: {epoch + 1}, Time: {time.time() - epoch_time:.2f}s | Train Loss: {train_loss:.7f} Val Loss: {val_loss:.7f}"
        )

        early_stopping(val_loss, model, checkpoint_dir)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        adjust_learning_rate(optimizer, epoch + 1, lr)

    print(
        f"Saving final checkpoint to {os.path.join(checkpoint_dir, 'checkpoint.pth')}"
    )
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "checkpoint.pth"))
    return early_stopping.get_val_loss_min()

def validate(model, val_loader, device, criterion, pred_len):
    model.eval()
    total_loss = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            with autocast(device_type="cuda"):
                outputs = model(batch_x)
                loss = criterion(outputs[:, -pred_len:, :], batch_y[:, -pred_len:, :])
            total_loss.append(loss.item())
            del batch_x, batch_y, outputs, loss
            torch.cuda.empty_cache()
            gc.collect()
    return np.average(total_loss)

def test(model, test_loader, device, scaler, pred_len):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            with autocast(device_type="cuda"):
                outputs = model(batch_x)
            outputs = outputs[:, -pred_len:, :].cpu().numpy()
            batch_y = batch_y[:, -pred_len:, :].cpu().numpy()
            preds.append(outputs)
            trues.append(batch_y)
            del batch_x, batch_y, outputs
            torch.cuda.empty_cache()
            gc.collect()

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    mae = MAE(preds, trues)
    mse = MSE(preds, trues)
    rse = RSE(preds, trues)
    return mae, mse, rse