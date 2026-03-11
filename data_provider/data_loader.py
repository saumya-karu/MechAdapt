
from typing import Union

import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from util.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


def get_alldata(filename='electricity.csv', root_path='./'):
    path = os.path.join(root_path, filename)
    if filename.endswith('.csv'):
        df = pd.read_csv(path)
        if filename.startswith('wind'):
            df['date'] = pd.date_range(start='2000-01-01', periods=len(df), freq='H')
    else:
        if filename.startswith('nyc'):
            import h5py
            x = h5py.File(path, 'r')
            data = list()
            for key in x.keys():
                data.append(x[key][:])
            ts = np.stack(data, axis=1)
            ts = ts.reshape((ts.shape[0], ts.shape[1] * ts.shape[2]))
            df = pd.DataFrame(ts)
            df['date'] = pd.date_range(start='2007-04-01', periods=len(df), freq='30T')
        elif filename.endswith('.npz'):
            ts = np.load(path)['data'].astype(np.float32)
            ts = ts.reshape((ts.shape[0], ts.shape[1] * ts.shape[2]))
            df = pd.DataFrame(ts)
            if filename == 'PeMSD4':
                df['date'] = pd.date_range(start='2017-07-01', periods=len(df), freq='5T')
            else:
                df['date'] = pd.date_range(start='2012-03-01', periods=len(df), freq='5T')
        elif filename.endswith('.h5'):
            df = pd.read_hdf(path)
            df['date'] = df.index.values
        elif filename.endswith('.txt'):
            df = pd.read_csv(path, header=None)
            df['date'] = pd.date_range(start='1/1/2007', periods=len(df), freq='10T')
        df = df[[df.columns[-1]] + list(df.columns[:-1])]
    return df

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv', ratio=None, borders=None,
                 target='OT', scale=True, timeenc=0, freq='h',
                 train_only=False, border=None):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features      # 'M', 'MS', 'S'
        self.target = target          # e.g. 'OT'
        self.target_idx = None        # will be set
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.border = border
        self.borders = borders

        self.root_path = root_path
        self.data_path = data_path

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # borders
        if self.borders is None:
            border1s = [
                0,
                12 * 30 * 24 - self.seq_len,
                12 * 30 * 24 + 4 * 30 * 24 - self.seq_len
            ]
            border2s = [
                12 * 30 * 24,
                12 * 30 * 24 + 4 * 30 * 24,
                12 * 30 * 24 + 8 * 30 * 24
            ]
        else:
            border1s, border2s = self.borders

        self.borders = (border1s, border2s)

        if self.border is None:
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
        else:
            border1, border2 = self.border

        # choose columns & target_idx
        if self.features in ['M', 'MS']:
            cols_data = list(df_raw.columns[1:])  # all vars except 'date'
            df_data = df_raw[cols_data]
            self.target_idx = cols_data.index(self.target)
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
            self.target_idx = 0
        else:
            raise ValueError(f"Unknown features mode: {self.features}")

        # scale
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        data = data.astype(np.float32)

        # time features
        df_stamp = df_raw[['date']][border1:border2].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        data_stamp = time_features(df_stamp, self.timeenc, freq=self.freq).astype(np.float32)

        self.data = data[:border2s[-1]]
        self.border = (border1, border2)

        # IMPORTANT: For Proceed, always keep Y multivariate
        self.data_x = data[border1:border2]     # [T, C or 1]
        self.data_y = data[border1:border2]     # [T, C or 1]
        self.data_stamp = data_stamp

        self.data_x = torch.from_numpy(self.data_x).float()
        self.data_y = torch.from_numpy(self.data_y).float()
        self.data_stamp = torch.from_numpy(self.data_stamp).float()

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[s_end:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# class Dataset_ETT_hour(Dataset):
#     def __init__(self, root_path, flag='train', size=None,
#                  features='S', data_path='ETTh1.csv', ratio=None, borders=None,
#                  target='OT', scale=True, timeenc=0, freq='h', train_only=False, border=None):
#         # size [seq_len, label_len, pred_len]
#         # info
#         if size == None:
#             self.seq_len = 24 * 4 * 4
#             self.label_len = 24 * 4
#             self.pred_len = 24 * 4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]
#
#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.timeenc = timeenc
#         self.freq = freq
#         self.border = border
#         self.borders = borders
#
#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()
#
#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         df_raw = pd.read_csv(os.path.join(self.root_path,
#                                           self.data_path))
#
#         if self.borders is None:
#             border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
#             border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
#         else:
#             border1s, border2s = self.borders
#         self.borders = (border1s, border2s)
#         if self.border is None:
#             border1 = border1s[self.set_type]
#             border2 = border2s[self.set_type]
#         else:
#             border1, border2 = self.border
#
#         if self.features == 'M':
#             cols_data = df_raw.columns[1:]
#             df_data = df_raw[cols_data]
#             self.target_idx = cols_data.index(self.target)
#         elif self.features == 'S':
#             df_data = df_raw[[self.target]]
#             self.target_idx = 0
#         elif self.features == 'MS':
#             cols_data = list(df_raw.columns[1:])
#             self.data_y = data[:, -1][border1:border2]
#             self.target_idx = cols_data.index(self.target)
#         else:
#             self.data_y = data[border1:border2]
#
#         if self.scale:
#             train_data = df_data[border1s[0]:border2s[0]]
#             self.scaler.fit(train_data.values)
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values
#
#         df_stamp = df_raw[['date']][border1:border2]
#         df_stamp['date'] = pd.to_datetime(df_stamp.date)
#         data_stamp = time_features(df_stamp, self.timeenc, freq=self.freq)
#
#         self.data = data[:border2s[-1]].astype(np.float32)
#         self.border = (border1, border2)
#
#         self.data_x = data[border1:border2]
#         self.data_y = data[border1:border2]
#         self.data_stamp = data_stamp.astype(np.float32)
#         self.data_x = torch.from_numpy(self.data_x).float()
#         self.data_y = torch.from_numpy(self.data_y).float()
#         self.data_stamp = torch.from_numpy(self.data_stamp).float()
#
#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len
#
#         seq_x = self.data_x[s_begin:s_end]
#         seq_y = self.data_y[s_end:r_end]
#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[r_begin:r_end]
#
#         return seq_x, seq_y, seq_x_mark, seq_y_mark
#
#     def __len__(self):
#         return len(self.data_x) - self.seq_len - self.pred_len + 1
#
#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv', ratio=None, borders=None,
                 target='OT', scale=True, timeenc=0, freq='t', train_only=False, border=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.border = border
        self.borders = borders

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        if self.borders is None:
            border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        else:
            border1s, border2s = self.borders
        self.borders = (border1s, border2s)
        if self.border is None:
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
        else:
            border1, border2 = self.border

            # choose columns & target_idx
        if self.features in ['M', 'MS']:
            cols_data = list(df_raw.columns[1:])  # all vars except 'date'
            df_data = df_raw[cols_data]
            self.target_idx = cols_data.index(self.target)
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
            self.target_idx = 0
        else:
            raise ValueError(f"Unknown features mode: {self.features}")

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, self.timeenc, freq=self.freq)

        self.data = data[:border2s[-1]].astype(np.float32)
        self.border = (border1, border2)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp.astype(np.float32)
        self.data_x = torch.from_numpy(self.data_x).float()
        self.data_y = torch.from_numpy(self.data_y).float()
        self.data_stamp = torch.from_numpy(self.data_stamp).float()

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[s_end:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
##########################################################################################################################
# import numpy as np
# import pandas as pd
# import torch
# from torch.utils.data import Dataset
# from sklearn.preprocessing import StandardScaler
#
# # Assumes these exist in your codebase (as in your snippet)
# # - get_alldata(data_path, root_path)
# # - time_features(df_stamp, timeenc, freq)
#
# class Dataset_Custom(Dataset):
#     """
#     Custom dataset with explicit contiguous splits:
#       - Train  = first 10% of the series
#       - Val    = next  5% immediately after train
#       - Test   = next 25% immediately after val
#       - Ignore remaining 60%
#
#     Notes:
#     - For val/test, we include an extra `seq_len` history BEFORE the split start
#       (i.e., border1 = split_start - seq_len) so the first window has enough past context.
#     - Scaling (StandardScaler) is fit ONLY on the *train* region (true train time span),
#       then applied to all used data.
#     - `__len__` counts available sliding windows in the selected border range.
#     """
#
#     def __init__(
#         self,
#         root_path,
#         flag="train",
#         size=None,
#         features="S",
#         data_path="ETTh1.csv",
#         ratio=None,          # kept for compatibility; not used unless you want to override defaults
#         borders=None,        # kept for compatibility (not used here)
#         target="OT",
#         scale=True,
#         timeenc=0,
#         freq="h",
#         train_only=False,
#         border=None,         # if provided, overrides the computed borders
#         split3=(0.60, 0.05, 0.25),  # (train, val, test) fractions; remainder ignored
#     ):
#         # size [seq_len, label_len, pred_len]
#         if size is None:
#             self.seq_len = 24 * 4 * 4
#             self.label_len = 24 * 4
#             self.pred_len = 24 * 4
#         else:
#             self.seq_len, self.label_len, self.pred_len = size
#
#         assert flag in ["train", "val", "test"]
#         self.flag = flag
#         type_map = {"train": 0, "val": 1, "test": 2}
#         self.set_type = type_map[flag]
#
#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.timeenc = timeenc
#         self.freq = freq
#         self.train_only = train_only
#
#         # User override border, if passed from outside
#         self.border = border
#
#         # Split config: train/val/test fractions for contiguous blocks, remainder ignored
#         self.split3 = split3
#
#         self.root_path = root_path
#         self.data_path = data_path
#
#         self.__read_data__()
#
#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         df_raw = get_alldata(self.data_path, self.root_path)
#
#         # Columns: ['date', ...(other features), target feature]
#         cols = list(df_raw.columns)
#         if self.features == "S":
#             cols.remove(self.target)
#         cols.remove("date")
#
#         N = len(df_raw)
#
#         # ---- Explicit split lengths (ignore remainder) ----
#         train_r, val_r, test_r = self.split3
#
#         # If train_only, keep behavior similar to original: use all available used region as train
#         # (but here "used region" is still bounded by split3 unless you change it)
#         if self.train_only:
#             # Use everything up to end of test block as "train"
#             num_train = int(N * (train_r + val_r + test_r))
#             num_vali = 0
#             num_test = 0
#             used_end = num_train
#         else:
#             num_train = int(N * train_r)
#             num_vali = int(N * val_r)
#             num_test = int(N * test_r)
#             used_end = num_train + num_vali + num_test  # stop here; ignore remainder
#
#         # Safety clamps
#         used_end = min(used_end, N)
#         num_train = min(num_train, used_end)
#         num_vali = min(num_vali, max(0, used_end - num_train))
#         num_test = min(num_test, max(0, used_end - num_train - num_vali))
#
#         # Split boundaries in timestep indices (true split starts, no seq_len padding)
#         train_start, train_end = 0, num_train
#         val_start, val_end = train_end, train_end + num_vali
#         test_start, test_end = val_end, val_end + num_test
#
#         # ---- Dataset window borders (with seq_len history for val/test) ----
#         # These are the borders actually used to slice data_x/data_y
#         border1s = [
#             train_start,
#             max(0, val_start - self.seq_len),
#             max(0, test_start - self.seq_len),
#         ]
#         border2s = [
#             train_end,
#             val_end,
#             test_end,
#         ]
#         self.borders = (border1s, border2s)
#
#         # Allow full override if an explicit border was passed in
#         if self.border is not None:
#             border1, border2 = self.border
#         else:
#             border1 = border1s[self.set_type]
#             border2 = border2s[self.set_type]
#
#         # Select feature matrix
#         if self.features in ["M", "MS"]:
#             cols_data = list(df_raw.columns[1:])  # all vars except 'date'
#             df_data = df_raw[cols_data]
#             self.target_idx = cols_data.index(self.target)
#         elif self.features == "S":
#             df_data = df_raw[[self.target]]
#             self.target_idx = 0
#         else:
#             raise ValueError(f"Unknown features mode: {self.features}")
#
#         # Fit scaler on TRUE TRAIN region only (no padding, no leakage)
#         if self.scale:
#             train_data = df_data[train_start:train_end]
#             self.scaler.fit(train_data.values)
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values
#
#         # Build time features for the *selected* slice (border1:border2)
#         df_stamp = df_raw[["date"]][border1:border2].copy()
#         df_stamp["date"] = pd.to_datetime(df_stamp["date"])
#         data_stamp = time_features(df_stamp, self.timeenc, freq=self.freq)
#
#         data = data.astype(np.float32)
#         self.data = data
#         self.border = (border1, border2)
#
#         # Slice tensors
#         self.data_x = data[border1:border2]  # [T, C] or [T, 1]
#         self.data_y = data[border1:border2]  # [T, C] or [T, 1]
#         self.data_stamp = data_stamp.astype(np.float32)
#
#         self.data_x = torch.from_numpy(self.data_x).float()
#         self.data_y = torch.from_numpy(self.data_y).float()
#         self.data_stamp = torch.from_numpy(self.data_stamp).float()
#
#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len
#
#         seq_x = self.data_x[s_begin:s_end]
#         seq_y = self.data_y[s_end:r_end]
#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[r_begin:r_end]
#
#         return seq_x, seq_y, seq_x_mark, seq_y_mark
#
#     def __len__(self):
#         # Number of available windows in this split slice.
#         # If too short, return 0 (avoid negative lengths).
#         L = len(self.data_x) - self.seq_len - self.pred_len + 1
#         return max(0, L)
#
#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv', ratio=None, borders=None,
                 target='OT', scale=True, timeenc=0, freq='h', train_only=False, border=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_only = train_only

        self.border = border
        self.split_ratio = (0.7, 0.2) if ratio is None else ratio
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = get_alldata(self.data_path, self.root_path)

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        if self.features == 'S':
            cols.remove(self.target)
        cols.remove('date')
        # print(cols)
        num_train = int(len(df_raw) * (self.split_ratio[0] if not self.train_only else 1))
        num_test = int(len(df_raw) * self.split_ratio[1])
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        self.borders = (border1s, border2s)
        if self.border is not None:
            border1, border2 = self.border
        else:
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

        if self.features in ['M', 'MS']:
            cols_data = list(df_raw.columns[1:])  # all vars except 'date'
            df_data = df_raw[cols_data]
            self.target_idx = cols_data.index(self.target)
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
            self.target_idx = 0
        else:
            raise ValueError(f"Unknown features mode: {self.features}")

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, self.timeenc, freq=self.freq)

        data = data.astype(np.float32)
        self.data = data
        self.border = (border1, border2)


        self.data_x = data[border1:border2]  # [T, C or 1]
        self.data_y = data[border1:border2]  # [T, C or 1]
        self.data_stamp = data_stamp.astype(np.float32)
        self.data_x = torch.from_numpy(self.data_x).float()
        self.data_y = torch.from_numpy(self.data_y).float()
        self.data_stamp = torch.from_numpy(self.data_stamp).float()

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[s_end:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_hour_CI(Dataset_ETT_hour):
    def __init__(self, root_path, *args, **kwargs):
        super().__init__(root_path, *args, **kwargs)
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = super().__len__()

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[s_end:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return super().__len__() * self.enc_in


class Dataset_ETT_minute_CI(Dataset_ETT_minute):
    def __init__(self, root_path, *args, **kwargs):
        super().__init__(root_path, *args, **kwargs)
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = super().__len__()

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[s_end:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return super().__len__() * self.enc_in


class Dataset_Custom_CI(Dataset_Custom):
    def __init__(self, root_path, *args, **kwargs):
        super().__init__(root_path, *args, **kwargs)
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = super().__len__()

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[s_end:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return super().__len__() * self.enc_in


class Dataset_Recent(Dataset):
    def __init__(self, dataset, gap: Union[int, tuple, list], recent_num=1, take_post=0, strength=0, **kwargs):
        super().__init__()
        self.more = gap - recent_num + 1
        self.dataset = dataset
        self.gap = gap
        self.recent_num = recent_num
        if strength:
            print("Modify time series with strength =", strength)
            for i in range(3, len(self.dataset.data_y)):
                self.dataset.data_x[i] *= 1 + 0.1 * (i // 24 % strength)

    def _stack(self, data):
        if isinstance(data[0], np.ndarray):
            return np.vstack(data)
        else:
            return torch.stack(data, 0)

    def __getitem__(self, index):
        if self.recent_num == 1:
            return self.dataset[index], self.dataset[index + self.gap]
        else:
            current_data = self.dataset[index + self.gap + self.recent_num - 1]
            if not isinstance(current_data, tuple):
                recent_data = tuple(self.dataset[index + n] for n in range(self.recent_num))
                recent_data = self._stack(recent_data)
                return current_data, recent_data
            else:
                recent_data = tuple([] for _ in range(len(current_data)))
                for past in range(self.recent_num):
                    for j, past_data in enumerate(self.dataset[index + past]):
                        recent_data[j].append(past_data)
                recent_data = tuple(self._stack(recent_d) for recent_d in recent_data)
            return recent_data, current_data

    def __len__(self):
        return len(self.dataset) - self.more
