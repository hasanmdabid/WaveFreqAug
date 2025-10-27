import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from PyEMD import EMD


# EMD decomposition
def emd_augment(data, sequence_length, n_IMF=10):
    n_imf, channel_num = n_IMF, data.shape[1]
    max_imf = 0
    # First, determine max_imf across all channels
    imf_counts = []
    for ci in range(channel_num):
        s = data[:, ci]
        IMF = EMD().emd(s)
        imf_counts.append(len(IMF) if len(IMF) > 0 else 1)
    max_imf = min(max(imf_counts), n_imf)  # Use minimum of max IMFs and n_imf
    print(f"max_imf: {max_imf}, n_imf: {n_imf}, channel_num: {channel_num}")

    # Initialize emd_data with max_imf
    emd_data = np.zeros((max_imf, data.shape[0], channel_num))
    for ci in range(channel_num):
        s = data[:, ci]
        IMF = EMD().emd(s)
        r_s = np.zeros((max_imf, data.shape[0]))
        if len(IMF) > 0:
            for i in range(min(len(IMF), max_imf)):
                r_s[i] = IMF[len(IMF) - 1 - i]
        else:
            r_s[0] = s  # Fallback to original signal if no IMFs
        emd_data[:, :, ci] = r_s[:max_imf, :]

    # Create sliding windows
    num_samples = len(data) - sequence_length + 1
    train_data_new = np.zeros((num_samples, max_imf, sequence_length, channel_num))
    for i in range(num_samples):
        train_data_new[i] = emd_data[:, i : i + sequence_length, :]
    print(f"train_data_new shape: {train_data_new.shape}")
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
