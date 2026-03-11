import time

from torch.utils.data import DataLoader
from tqdm import tqdm

import settings
from data_provider.data_factory import data_provider, get_dataset
from exp import Exp_Online
import torch.nn.functional as F

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import warnings
import copy
from util.metrics import update_metrics, calculate_metrics
from util.tools import test_params_flop

warnings.filterwarnings('ignore')

def get_period(dataset_name):
    if "ETTh1" in dataset_name:
        period = 24
    elif "ETTh2" in dataset_name:
        period = 24
    elif "ETTh2_perbuted" in dataset_name:
        period = 24
    elif "ETTm1" in dataset_name:
        period = 96
    elif "ETTm1_perbuted" in dataset_name:
        period = 96
    elif "ETTm2" in dataset_name:
        period = 96
    elif "electricity" in dataset_name:
        period = 24
    elif "ECL" in dataset_name:
        period = 24
    elif "traffic" in dataset_name.lower():
        period = 24
    elif "traffic_perbuted" in dataset_name.lower():
        period = 24
    elif "illness" in dataset_name.lower():
        period = 52.142857
    elif "weather" in dataset_name.lower():
        period = 144
    elif "weather_perbuted" in dataset_name.lower():
        period = 144
    elif "Exchange" in dataset_name:
        period = 1
    elif "WTH_informer" in dataset_name:
            period = 24
    else:
        period = 1
    return period


class Exp_SOLID(Exp_Online):
    def __init__(self, args):
        super(Exp_SOLID, self).__init__(args)
        self.rep_path = args.rep_path if hasattr(args, 'use_rep') and args.use_pred and hasattr(args, 'rep_path') else None
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.label_len = args.label_len # Transformer
        self.test_train_num = args.test_train_num
        if not args.whole_model:
            self.model.requires_grad_(False)
        else:
            for j in range(len(self.model_optim.param_groups)):
                self.model_optim.param_groups[j]['lr'] = args.online_learning_rate
        self.period = get_period(self.args.dataset)
        linear_map = {
                "Linear": "Linear",
                "NLinear": "Linear",
                "PatchTST": "model.head.linear",
                "TCN": "regressor",
                "iTransformer": "projector",
                "default": "decoder.projection",
        }
        self.linear_name = linear_map[self.args.model] if self.args.model in linear_map else linear_map["default"]
        if self.args.normalization:
            self.linear_name = 'backbone.' + self.linear_name
        if self.args.whole_model:
            self.final_head = self.model
        else:
            self.final_head = self.model.get_submodule(self.linear_name)
            self.final_head.requires_grad_(True)
        indices = torch.arange(self.pred_len + self.test_train_num - 1, self.pred_len - 1, step=-1) % self.period
        threshold = self.period * self.args.lambda_period
        self.indices = torch.arange(self.test_train_num)[(indices <= threshold) & (indices >= -threshold)]
        self.indices_x = (self.indices.unsqueeze(-1) + torch.arange(self.seq_len)).unsqueeze(-1).expand(-1, -1, self.args.enc_in)
        self.indices_y = (self.indices.unsqueeze(-1) + torch.arange(self.seq_len, self.seq_len + self.pred_len)).unsqueeze(-1).expand(-1, -1, self.args.enc_in)
        self.indices_x_mark = (self.indices.unsqueeze(-1) + torch.arange(self.seq_len)).unsqueeze(-1)
        self.indices_y_mark = (self.indices.unsqueeze(-1) + torch.arange(self.seq_len - self.label_len, self.seq_len + self.pred_len)).unsqueeze(-1)

        self.indices = self.indices.to(self.device)
        self.indices_x = self.indices_x.to(self.device)
        self.indices_x_mark = self.indices_x_mark.to(self.device)
        self.indices_y_mark = self.indices_y_mark.to(self.device)
        self.indices_y = self.indices_y.to(self.device)

        self.batch_size = -1
        if self.args.dataset == 'Traffic' and self.args.selected_data_num > 20:
            self.batch_size = 20
        elif self.args.dataset == 'ECL' and self.args.selected_data_num > 40:
            self.batch_size = 20


    def _select_optimizer(self, *args, **kwargs):
        if self.args.whole_model:
            return super()._select_optimizer(*args, **kwargs)
        return None

    def _forward(self, x, rep):
        pred = self.final_head(rep.detach())
        if self.model == 'PatchTST' and self.args.revin:
            pred = pred.permute(0,2,1)
            pred = self.model.backbone.revin_layer(pred, 'denorm')
        elif self.model == 'iTransformer':
            means = x.mean(1, keepdim=True).detach()
            stdev = torch.sqrt(torch.var(x - means, dim=1, keepdim=True, unbiased=False) + 1e-5)
            pred = pred * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            pred = pred + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        elif self.model == 'TCN':
            pred = pred.reshape(len(pred), self.pred_len, -1)
        if self.args.normalization.lower() == 'revin':
            pred = self.model.processor(pred, 'denorm')
        return pred

    def online(self, online_data=None, target_variate=100000, phase='test', show_progress=False):
        predictions = []
        self.model.eval()
        if online_data is None:
            online_data = get_dataset(self.args, 'test', self.device,
                                      take_pre=self.test_train_num + self.pred_len - 1,
                                      wrap_class=self.args.wrap_data_class, **self.wrap_data_kwargs)
        assert self.args.borders[1][0] - self.test_train_num - self.pred_len + 1 >= 0
        if self.rep_path:
            all_reps = np.load(self.rep_path)
            assert len(all_reps) == self.args.borders[1][1] - self.seq_len - self.pred_len + 1
            all_reps = all_reps[self.args.borders[1][0] - self.test_train_num - self.pred_len + 1:]
        data_x = online_data.data_x
        data_y = online_data.data_y
        data_ts = online_data.data_stamp.to(data_x.device)
        if self.indices_x_mark.shape[-1] != data_ts.shape[-1]:
            self.indices_x_mark = self.indices_x_mark.expand(-1, -1, data_ts.shape[-1])
            self.indices_y_mark = self.indices_y_mark.expand(-1, -1, data_ts.shape[-1])
        test_loader = DataLoader(online_data, batch_size=1, shuffle=False, num_workers=self.args.num_workers,
                                 drop_last=False, pin_memory=False)
        criterion = nn.MSELoss()
        statistics = {k: 0 for k in ['total', 'y_sum', 'MSE', 'MAE', 'MAPE', "SMAPE"]}
        if not self.args.continual:
            if not self.args.whole_model:
                pretrained_state_dict = copy.deepcopy(self.final_head.state_dict())
            else:
                pretrained_state_dict = copy.deepcopy(self.state_dict())
        for i, batch in enumerate(tqdm(test_loader, mininterval=10)):
            if i < self.test_train_num + self.pred_len - 1:
                continue
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            batch_x = batch_x.to(self.device)

            t = i + self.seq_len
            start = i - self.pred_len - self.test_train_num + 1
            lookback_x = data_x[start: t].expand(len(self.indices), -1, -1).gather(1, self.indices_x)

            # calculate sample similarity
            distance_pairs = F.pairwise_distance(batch_x.view(-1), lookback_x.view(len(lookback_x), -1), p=2)
            selected_indices = distance_pairs.topk(self.args.selected_data_num, largest=False)[1]
            idx = self.indices[selected_indices]
            selected_x = lookback_x[selected_indices]
            selected_y = data_y[start: t].expand(len(idx), -1, -1).gather(1, self.indices_y[selected_indices])
            if self.rep_path:
                selected_reps = all_reps[i - (self.test_train_num + self.pred_len - 1) + idx]
                # selected_pred = self._forward(selected_x, selected_reps)
            else:
                selected_x_mark = data_ts[start: t].expand(len(idx), -1, -1).gather(1, self.indices_x_mark[selected_indices])
                selected_y_mark = data_ts[start: t].expand(len(idx), -1, -1).gather(1, self.indices_y_mark[selected_indices])

            # if self.args.continual:
            self.model.train()
            if not self.args.whole_model:
                selected_pred = self.forward([selected_x, selected_y, selected_x_mark, selected_y_mark])
                if isinstance(selected_pred, tuple):
                    selected_pred = selected_pred[0]
                pretrained_state_dict = copy.deepcopy(self.final_head.state_dict())
                model_optim = optim.SGD(self.final_head.parameters(), lr=self.args.online_learning_rate)

                loss = criterion(selected_pred, selected_y)
                loss.backward()
                model_optim.step()
                model_optim.zero_grad()
            else:
                self._update([selected_x, selected_y, selected_x_mark, selected_y_mark], criterion, self.model_optim)
                self.model_optim.zero_grad()

            with torch.no_grad():
                self.model.eval()
                outputs = self.forward(batch)
                if self.args.do_predict:
                    if isinstance(outputs, (tuple, list)):
                        outputs = outputs[0]
                    predictions.append(outputs.detach().cpu().numpy())
                true = batch[self.label_position]
                if not self.args.pin_gpu:
                    true = true.to(self.device)
                target_variate = getattr(self.args, "target_variate")
                update_metrics(outputs, true, statistics, target_variate)
            if not self.args.continual:
                if not self.args.whole_model:
                    self.final_head.load_state_dict(pretrained_state_dict)
                else:
                    self.load_state_dict(pretrained_state_dict)
        metrics = calculate_metrics(statistics)
        mse, mae, rmse, smape = metrics['MSE'], metrics['MAE'], metrics['RMSE'], metrics['SMAPE']
        print('rmse:{}, mae:{}, smape:{}'.format(rmse, mae, smape))

        if self.args.do_predict:
            return rmse, mae, smape, online_data, predictions
        else:
            return rmse, mae, smape,  online_data


    def analysis_online(self):
        self.model.eval()
        online_data = get_dataset(self.args, 'test', self.device,
                                  take_pre=self.test_train_num + self.pred_len - 1,
                                  wrap_class=self.args.wrap_data_class, **self.wrap_data_kwargs)
        assert self.args.borders[1][0] - self.test_train_num - self.pred_len + 1 >= 0
        data_x = online_data.data_x.to(self.device)
        data_y = online_data.data_y.to(self.device)
        data_ts = online_data.data_stamp.to(data_x.device)
        if self.indices_x_mark.shape[-1] != data_ts.shape[-1]:
            self.indices_x_mark = self.indices_x_mark.expand(-1, -1, data_ts.shape[-1])
            self.indices_y_mark = self.indices_y_mark.expand(-1, -1, data_ts.shape[-1])
        test_loader = DataLoader(online_data, batch_size=1, shuffle=False, num_workers=self.args.num_workers,
                                 drop_last=False, pin_memory=False)
        criterion = nn.MSELoss()
        statistics = {k: 0 for k in ['total', 'y_sum', 'MSE', 'MAE', 'SMAPE']}
        if not self.args.continual:
            if not self.args.whole_model:
                pretrained_state_dict = copy.deepcopy(self.final_head.state_dict())
            else:
                pretrained_state_dict = copy.deepcopy(self.state_dict())

        times_update = []
        times_infer = []
        print('GPU Mem:', torch.cuda.max_memory_allocated())

        j = 0
        for i, batch in enumerate(tqdm(test_loader, mininterval=10)):

            if i < self.test_train_num + self.pred_len - 1:
                continue

            # if i == 0:
            #     print('New GPU Mem:', torch.cuda.memory_allocated())
            # if j > 10:
            #     times_infer.append(time.time() - start_time)
            if j == 50:
                break

            start_time = time.time()
            batch = [d.to(self.device) for d in batch]
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch

            t = i + self.seq_len
            start = i - self.pred_len - self.test_train_num + 1
            lookback_x = data_x[start: t].expand(len(self.indices), -1, -1).gather(1, self.indices_x)

            # calculate sample similarity
            distance_pairs = F.pairwise_distance(batch_x.view(-1), lookback_x.view(len(lookback_x), -1), p=2)
            selected_indices = distance_pairs.topk(self.args.selected_data_num, largest=False)[1]
            idx = self.indices[selected_indices]
            selected_x = lookback_x[selected_indices]
            selected_y = data_y[start: t].expand(len(idx), -1, -1).gather(1, self.indices_y[selected_indices])
            selected_x_mark = data_ts[start: t].expand(len(idx), -1, -1).gather(1, self.indices_x_mark[selected_indices])
            selected_y_mark = data_ts[start: t].expand(len(idx), -1, -1).gather(1, self.indices_y_mark[selected_indices])

            if j == 5:
                print(selected_x.shape)
            # if self.args.continual:
            self.model.train()
            if not self.args.whole_model:
                pretrained_state_dict = copy.deepcopy(self.final_head.state_dict())
                model_optim = optim.SGD(self.final_head.parameters(), lr=self.args.online_learning_rate)
                for ii in range(len(selected_x)):
                    selected_pred = self.forward([selected_x[[ii]], selected_y[[ii]], selected_x_mark[[ii]], selected_y_mark[[ii]]])
                    if isinstance(selected_pred, tuple):
                        selected_pred = selected_pred[0]
                    loss = criterion(selected_pred, selected_y[[ii]])
                    loss.backward()
                model_optim.step()
                model_optim.zero_grad()
            else:
                # self._update([selected_x, selected_y, selected_x_mark, selected_y_mark], criterion, self.model_optim)
                # self.model_optim.zero_grad()
                for ii in range(len(selected_x)):
                    selected_pred = self.forward([selected_x[[ii]], selected_y[[ii]], selected_x_mark[[ii]], selected_y_mark[[ii]]])
                    if isinstance(selected_pred, tuple):
                        selected_pred = selected_pred[0]
                    loss = criterion(selected_pred, selected_y[[ii]])
                    loss.backward()
                self.model_optim.step()
                self.model_optim.zero_grad()

            if j > 10:
                times_update.append(time.time() - start_time)

            with torch.no_grad():
                start_time = time.time()
                self.model.eval()
                outputs = self.forward(batch)
                if j > 10:
                    times_infer.append(time.time() - start_time)
                if j == 50:
                    break
            j += 1

            if not self.args.continual:
                if not self.args.whole_model:
                    self.final_head.load_state_dict(pretrained_state_dict)
                else:
                    self.load_state_dict(pretrained_state_dict)

        print('Final GPU Mem:', torch.cuda.max_memory_allocated())
        times_update = (sum(times_update) - min(times_update) - max(times_update)) / (len(times_update) - 2)
        times_infer = (sum(times_infer) - min(times_infer) - max(times_infer)) / (len(times_infer) - 2)
        print('Update Time:', times_update)
        print('Infer Time:', times_infer)
        print('Latency:', times_update + times_infer)
        test_params_flop(self.model, (1, self.args.seq_len, self.args.enc_in))

    def predict(self, path, setting, load=False):

        import os, numpy as np, pandas as pd, matplotlib.pyplot as plt

        if load:
            print('Loading', path)
            self.load_checkpoint(path)


        if hasattr(self, "update_valid"):
            self.update_valid()

        ret = self.online()
        predictions = None
        if isinstance(ret, (tuple, list)):
            if len(ret) == 4:
                _, _, _, predictions = ret

        preds_seq = []
        if predictions:
            preds_seq = [np.asarray(p) for p in predictions]
        if not preds_seq and getattr(self, 'pred_list', None):
            preds_seq = [np.asarray(p) for p in self.pred_list]
        if not preds_seq:
            raise RuntimeError("No predictions captured. Ensure online() fills 'predictions' or 'self.pred_list'.")

        preds_arr = np.vstack(preds_seq)

        if not getattr(self, 'true_list', None):
            raise RuntimeError("No ground truths captured. Ensure online() fills 'self.true_list'.")
        trues_arr = np.vstack([np.asarray(t) for t in self.true_list])

        def ensure_3d(x):
            x = np.asarray(x)
            if x.ndim == 1: return x[:, None, None]
            if x.ndim == 2: return x[:, :, None]
            return x

        preds_arr = ensure_3d(preds_arr)
        trues_arr = ensure_3d(trues_arr)

        assert preds_arr.shape[:2] == trues_arr.shape[:2], \
            f"temporal dims mismatch: {preds_arr.shape[:2]} vs {trues_arr.shape[:2]}"

        T, pred_len, C_pred = preds_arr.shape
        _, _, C_true = trues_arr.shape

        features_mode = getattr(self.args, 'features', 'M')


        target_idx = getattr(self.args, 'target_variate', None)
        if target_idx is None:
            target_idx = C_pred - 1
        if target_idx >= C_pred:
            print(f"[warn] target_variate={target_idx} >= C_pred={C_pred}; using 0")
            target_idx = 0

        if (features_mode == 'MS' and C_pred != 1) or (C_true == 1 and C_pred > 1):
            preds_arr = preds_arr[:, :, [target_idx]]
            C_pred = 1

        if C_pred == 1 and C_true > 1:
            trues_arr = trues_arr[:, :, [target_idx]]
            C_true = 1


        assert preds_arr.shape == trues_arr.shape, \
            f"shapes still mismatch after harmonization: {preds_arr.shape} vs {trues_arr.shape}"


        C = preds_arr.shape[2]


        target_idx = getattr(self.args, 'target_variate', 0)
        if target_idx is None:
            target_idx = 0
        if target_idx >= C:
            print(f"[warn] target_variate={target_idx} >= C={C}; using 0")
            target_idx = 0


        if features_mode == 'MS' and C != 1:
            preds_arr = preds_arr[:, :, [target_idx]]
            trues_arr = trues_arr[:, :, [target_idx]]
            C = 1


        save_dir = os.path.join('./results', setting)
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'preds.npy'), preds_arr)
        np.save(os.path.join(save_dir, 'trues.npy'), trues_arr)

        #
        ms_suffix_on = bool(getattr(self.args, 'ms_channel_suffix', True))
        one_index = bool(getattr(self.args, 'channel_one_indexed', True))
        chan_label = (target_idx + 1) if one_index else target_idx

        cols, data_cols = [], []
        if C == 1:
            ch_tag = f"_channel_{chan_label}" if (features_mode == 'MS' and ms_suffix_on) else ""
            for h in range(pred_len):
                cols.append(f"pred_h{h + 1}{ch_tag}")
                data_cols.append(preds_arr[:, h, 0])
            for h in range(pred_len):
                cols.append(f"true_h{h + 1}{ch_tag}")
                data_cols.append(trues_arr[:, h, 0])
        else:
            for c in range(C):
                for h in range(pred_len):
                    cols.append(f"pred_h{h + 1}_c{c}")
                    data_cols.append(preds_arr[:, h, c])
                for h in range(pred_len):
                    cols.append(f"true_h{h + 1}_c{c}")
                    data_cols.append(trues_arr[:, h, c])

        df = pd.DataFrame(np.column_stack(data_cols), columns=cols)
        csv_path = os.path.join(save_dir, 'preds_trues_wide.csv')
        df.to_csv(csv_path, index=False)
