import torch
import torch.nn as nn
import numpy as np
import os
import time
from src.aug_method import Augmentation


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


def adjust_learning_rate(optimizer, epoch, lr):
    lr_adjust = {epoch: lr * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust:
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print(f"Adjusted learning rate to {lr}")


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
    aug_type,
    aug_params,
    epochs=10,
    lr=0.01,
    patience=3,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    aug = Augmentation()
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    path = f"./checkpoints/{aug_type}"
    if not os.path.exists(path):
        os.makedirs(path)

    sampling_rate = aug_params.get("sampling_rate", 0.5)
    aug_params_clean = {k: v for k, v in aug_params.items() if k != "sampling_rate"}

    for epoch in range(epochs):
        model.train()
        train_loss = []
        epoch_time = time.time()
        for i, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            if aug_type == "Adaptive-Channel-Preserve":
                augmented_data = aug.adaptive_channel_preserve(
                    batch_x, batch_y[:, -pred_len:, :], **aug_params_clean
                )
            elif aug_type == "TFMix":
                augmented_data = aug.tf_mix(
                    batch_x, batch_y[:, -pred_len:, :], **aug_params_clean
                )
            elif aug_type == "STL-Mix":
                augmented_data = aug.stl_mix(
                    batch_x,
                    batch_y[:, -pred_len:, :],
                    device=device,
                    **aug_params_clean,
                )
            elif aug_type == "VMD-Mix":
                augmented_data = aug.vmd_mix(
                    batch_x,
                    batch_y[:, -pred_len:, :],
                    device=device,
                    **aug_params_clean,
                )
            batch_x2, batch_y2 = (
                augmented_data[:, :seq_len, :],
                augmented_data[:, seq_len : seq_len + label_len + pred_len, :],
            )
            sampling_steps = int(batch_x2.shape[0] * sampling_rate)
            indices = torch.randperm(batch_x2.shape[0], device=device)[:sampling_steps]
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
