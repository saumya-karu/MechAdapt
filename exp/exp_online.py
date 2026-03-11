import copy

import numpy as np
from tqdm import tqdm

from data_provider.data_factory import data_provider, get_dataset, get_dataloader
from data_provider.data_loader import Dataset_Recent
from exp.exp_main import Exp_Main
from models.OneNet import OneNet, Model_Ensemble
from util.buffer import Buffer
from util.metrics import metric, update_metrics, calculate_metrics
import torch
import torch.nn.functional as F
from torch import optim, nn
import torch.distributed as dist
import numpy as np

import os
import time

import warnings

from util.tools import test_params_flop

warnings.filterwarnings('ignore')

transformers = ['Autoformer', 'Transformer', 'Informer']

class Exp_Online(Exp_Main):
    def __init__(self, args):
        super().__init__(args)
        self.online_phases = ['test', 'online']
        self.wrap_data_kwargs.update(recent_num=1, gap=self.args.pred_len)

    def _get_data(self, flag, **kwargs):
        if flag in self.online_phases:
            if self.args.leakage:
                data_set = get_dataset(self.args, flag, self.device, wrap_class=self.args.wrap_data_class, **kwargs)
                data_loader = get_dataloader(data_set, self.args, 'online' if flag == 'test' else 'test')
            else:
                data_set = get_dataset(self.args, flag, self.device,
                                       wrap_class=self.args.wrap_data_class + [Dataset_Recent],
                                       **self.wrap_data_kwargs, **kwargs)
                data_loader = get_dataloader(data_set, self.args, 'online')
            return data_set, data_loader
        else:
            return super()._get_data(flag, **kwargs)

    def vali(self, vali_data, vali_loader, criterion):
        self.phase = 'val'
        if self.args.leakage or 'val' not in self.online_phases:
            mse = super().vali(vali_data, vali_loader, criterion)
        else:
            if self.args.local_rank <= 0:
                state_dict = copy.deepcopy(self.state_dict())
                target_variate = getattr(self.args, 'target_variate')
                mse = self.online(online_data=vali_data, target_variate=target_variate, phase='val')[0]
                if self.args.local_rank == 0:
                    mse = torch.tensor(mse, device=self.device)
                self.load_state_dict(state_dict, strict=not (hasattr(self.args, 'freeze') and self.args.freeze))
            else:
                mse = torch.tensor(0, device=self.device)
            if self.args.local_rank >= 0:
                dist.all_reduce(mse, op=dist.ReduceOp.SUM)
                mse = mse.item()
        return mse

    def update_valid(self, valid_data=None):
        self.phase = 'online'

        if hasattr(self.args, 'leakage') and self.args.leakage:
            if self.args.model == 'PatchTST':
                valid_data = get_dataset(self.args, 'val', self.device,
                                         wrap_class=self.args.wrap_data_class + [Dataset_Recent],
                                         take_post=self.args.pred_len - 1, **self.wrap_data_kwargs)
                self.online_information_leakage_PatchTST(valid_data, None, 'online', True)
            else:
                valid_data = get_dataset(self.args, 'val', self.device, wrap_class=self.args.wrap_data_class,
                                         take_pre=True, take_post=self.args.pred_len - 1, **self.wrap_data_kwargs)
                self.online_information_leakage(valid_data, None, 'online', True)
            return []

        if valid_data is None or not isinstance(valid_data, Dataset_Recent):
            valid_data = get_dataset(self.args, 'val', self.device,
                                     wrap_class=self.args.wrap_data_class + [Dataset_Recent],
                                     take_post=self.args.pred_len - 1, **self.wrap_data_kwargs)
        valid_loader = get_dataloader(valid_data, self.args, 'online')
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None
        self.model.train()
        predictions = []
        for i, (recent_batch, current_batch) in enumerate(tqdm(valid_loader, mininterval=10)):
            self._update_online(recent_batch, criterion, model_optim, scaler)
            if self.args.do_predict:
                self.model.eval()
                with torch.no_grad():
                    outputs = self.forward(current_batch)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                predictions.append(outputs.detach().cpu().numpy())
                self.model.train()
        return predictions

    def _update_online(self, batch, criterion, optimizer, scaler=None):
        if batch[0].dim() == 3:
            return self._update(batch, criterion, optimizer, scaler)
        else:
            batch = [b[0] for b in batch]
            if not isinstance(optimizer, tuple):
                optimizer = (optimizer,)
            for optim in optimizer:
                optim.zero_grad()
            outputs = self.forward(batch)
            batch_y = batch[1]
            if not self.args.pin_gpu:
                batch_y = batch_y.to(self.device)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            loss = 0
            H = batch_y.shape[1]
            for i in range(H):
                loss += criterion(outputs[i, :H-i], batch_y[i, :H-i])
            if self.args.use_amp:
                scaler.scale(loss).backward()
                for optim in optimizer:
                    scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                for optim in optimizer:
                    optim.step()

            return loss, outputs



    def online(self, online_data=None, target_variate=100000, phase='test', show_progress=False):
        import numpy as np

        self.pred_list = []  # each item: (B=1, H, C) or (1, H)
        self.true_list = []  # each item: (B=1, H, C)
        self.phase = phase
        target_variate = getattr(self.args, 'target_variate')
        if hasattr(self.args, 'leakage') and self.args.leakage:
            if self.args.model == 'PatchTST':
                return self.online_information_leakage_PatchTST(online_data, target_variate, phase, show_progress)
            else:
                return self.online_information_leakage(online_data, target_variate, phase, show_progress)
        if online_data is None:
            online_data = get_dataset(self.args, phase, self.device,
                                      wrap_class=self.args.wrap_data_class + [Dataset_Recent],
                                      **self.wrap_data_kwargs)
        # online_loader_initial = get_dataloader(online_data.dataset, self.args, flag='online')
        online_loader = get_dataloader(online_data, self.args, flag='online')

        if self.args.do_predict:
            predictions = []
            trues = []  # NEW
            self.pred_list = []  # NEW: clear previous runs
            self.true_list = []
        statistics = {k: 0 for k in ['total', 'y_sum', 'MSE', 'MAE', 'SMAPE']}

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        if phase == 'test' and hasattr(self.args, 'debug') and self.args.debug:
            import tensorboardX as tensorboard
            import shutil
            log_dir = f'run/{self.args.online_method}_{self.args.dataset}_{self.args.seq_len}_{self.args.pred_len}_' \
                      f'{self.args.learning_rate}_{self.args.online_learning_rate}_{self.args.trigger_threshold}_' \
                      f'{self.args.tune_mode}_' \
                      f'{self.args.bottleneck_dim}_{self.args.penalty}_{self.args.comment}/' \
                      f'{time.strftime("%Y%m%d%H%M", time.localtime())}'
            print(log_dir)
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir)
            self.writer = tensorboard.SummaryWriter(log_dir=log_dir)

        if phase == 'test' or show_progress:
            online_loader = tqdm(online_loader, mininterval=10)
        for i, (recent_data, current_data) in enumerate(online_loader):
            self.model.train()
            loss, _ = self._update_online(recent_data, criterion, model_optim, scaler)
            self.model.eval()
            with torch.no_grad():
                outputs = self.forward(current_data)
                y_true = current_data[self.label_position]

                # always work with the tensor (not tuple) for metrics
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]

                # --------- predictions saved for later (slice by channel) ----
                if self.args.do_predict:
                    out_np = outputs.detach().cpu().numpy()
                    true_np = y_true.detach().cpu().numpy()

                    # ensure 3D: (B, H, C)
                    if out_np.ndim == 2:  # (B, H)
                        out_np = out_np[:, :, None]
                    if true_np.ndim == 2:
                        true_np = true_np[:, :, None]

                    # choose target channel if requested
                    t_idx = getattr(self.args, 'target_variate')
                    if t_idx is not None and out_np.shape[-1] > 1:
                        t_idx = int(max(0, min(t_idx, out_np.shape[-1] - 1)))
                        out_np = out_np[..., t_idx:t_idx + 1]
                        true_np = true_np[..., t_idx:t_idx + 1]

                    predictions.append(out_np)
                    trues.append(true_np)


                update_metrics(outputs, y_true.to(self.device), statistics, target_variate)

                # --- drift score logging (safe, optional) ---
                if hasattr(self.model, "last_score") and self.model.last_score is not None:
                    if not hasattr(self, "drift_scores"):
                        self.drift_scores = []
                    self.drift_scores.append(self.model.last_score.item())
                if hasattr(self.model, "last_gate") and self.model.last_gate is not None:
                    if not hasattr(self, "gate_values"):
                        self.gate_values = []
                    self.gate_values.append(self.model.last_gate.item())

                # self.pred_list.append(out.detach().cpu().numpy())
                # self.true_list.append(true.detach().cpu().numpy())

                if phase == 'test' and hasattr(self.args, 'debug') and self.args.debug:
                    if isinstance(outputs, (tuple, list)):
                        outputs = outputs[0]
                    mse = F.mse_loss(outputs, current_data[self.label_position].to(self.device))
                    self.writer.add_scalar('Online/MSE', mse, i)
                    self.writer.add_scalar('Online/avg_MSE', statistics['MSE'] / statistics['total'], i)
                    # print('Online MSE: {:.2f}'.format(mse.item()))
                    # for j in range(current_data[self.label_position].shape[-1]):
                    #     self.writer.add_scalar(f'Online/x_{j}', current_data[self.label_position][0, 0, j], i)

        metrics = calculate_metrics(statistics)
        mse, mae, rmse, smape = metrics['MSE'], metrics['MAE'], metrics['RMSE'], metrics['SMAPE']
        if phase == 'test':
            print('rmse:{}, mae:{}, smape:{}'.format(rmse, mae, smape))
            #print("OUTPUT SHAPE:", outputs.shape)
        if phase == 'test':
            if hasattr(self, "drift_scores"):
                import numpy as np
                np.save(f"./results/{self.args.model_id}_{self.args.online_method}_drift_scores.npy",
                        np.array(self.drift_scores))
            if hasattr(self, "gate_values"):
                np.save(f"./results/{self.args.model_id}_gate_values.npy",
                        np.array(self.gate_values))
        if self.args.do_predict:
            # stash for predict()
            self.pred_list = predictions
            self.true_list = trues
            return rmse, mae, smape,  online_data, predictions
        else:
            return rmse, mae, smape, online_data

    def online_information_leakage(self, online_data=None, target_variate=100000, phase='test', show_progress=False):
        if online_data is None:
            online_data = get_dataset(self.args, phase, self.device, wrap_class=self.args.wrap_data_class,
                                      **self.wrap_data_kwargs)
        online_loader = get_dataloader(online_data, self.args, flag='online')

        if self.args.do_predict:
            predictions = []

        statistics = {k: 0 for k in ['total', 'y_sum', 'MSE', 'MAE', 'MAPE']}
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        if phase == 'test' or show_progress:
            online_loader = tqdm(online_loader, mininterval=10)
        self.model.train()
        for i, current_data in enumerate(online_loader):
            loss, outputs = self._update_online(current_data, criterion, model_optim, scaler)
            with torch.no_grad():
                target_variate = getattr(self.args, 'target_variate')
                update_metrics(outputs, current_data[self.label_position].to(self.device), statistics, target_variate)
            if self.args.do_predict:
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                predictions.append(outputs.detach().cpu().numpy())

        metrics = calculate_metrics(statistics)
        mse, mae, rmse, smape = metrics['MSE'], metrics['MAE'], metrics['RMSE'], metrics['SMAPE']
        if phase == 'test':
            print('rmse:{}, mae:{}, smape:{}'.format(rmse, mae
                                                    , smape))
        if self.args.do_predict:
            return rmse, mae, smape, online_data, predictions
        else:
            return rmse, mae, smape, online_data

    def online_information_leakage_PatchTST(self, online_data=None, target_variate=100000, phase='test', show_progress=False):

        self.phase = phase
        if online_data is None:
            online_data = get_dataset(self.args, phase, self.device, wrap_class=self.args.wrap_data_class + [Dataset_Recent],
                                      **self.wrap_data_kwargs)
        # online_loader_initial = get_dataloader(online_data.dataset, self.args, flag='online')
        online_loader = get_dataloader(online_data, self.args, flag='online')

        if self.args.do_predict:
            predictions = []
        statistics = {k: 0 for k in ['total', 'y_sum', 'MSE', 'MAE', 'MAPE']}

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        if phase == 'test' or show_progress:
            online_loader = tqdm(online_loader, mininterval=10)
        for i, (recent_data, current_data) in enumerate(online_loader):
            self.model.train()
            with torch.no_grad():
                outputs = self.forward(recent_data)
            self.model.eval()
            loss, outputs = self._update_online(current_data, criterion, model_optim, scaler)
            # assert not torch.isnan(loss)
            # with torch.no_grad():
                # outputs = self.forward(current_data)
            if self.args.do_predict:
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                predictions.append(outputs.detach().cpu().numpy())
            target_variate = getattr(self.args, 'target_variate')
            update_metrics(outputs, current_data[self.label_position].to(self.device), statistics, target_variate)

        metrics = calculate_metrics(statistics)
        mse, mae, rmse, smape = metrics['MSE'], metrics['MAE'], metrics['RMSE'], metrics['SMAPE']
        if phase == 'test':
            print('rmse:{}, mae:{}, smape:{}'.format(rmse, mae, smape))
        if self.args.do_predict:
            return rmse, mae, smape, online_data, predictions
        else:
            return rmse, mae, smape, online_data

    def analysis_online(self):
        online_data = get_dataset(self.args, 'test', self.device, wrap_class=self.args.wrap_data_class + [Dataset_Recent],
                                  **self.wrap_data_kwargs)
        online_loader = get_dataloader(online_data, self.args, flag='online')
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        times_update = []
        times_infer = []
        print('GPU Mem:', torch.cuda.max_memory_allocated())
        for i, (recent_data, current_data) in enumerate(online_loader):
            start_time = time.time()
            self.model.train()
            recent_data = [d.to(self.device) for d in recent_data]
            loss, _ = self._update_online(recent_data, criterion, model_optim, scaler)
            if i > 10:
                times_update.append(time.time() - start_time)
            self.model.eval()
            with torch.no_grad():
                start_time = time.time()
                current_data = [d.to(self.device) for d in current_data]
                self.forward(current_data)
            # if i == 0:
            #     print('New GPU Mem:', torch.cuda.memory_allocated())
            if i > 10:
                times_infer.append(time.time() - start_time)
            if i == 50:
                break
        print('Final GPU Mem:', torch.cuda.max_memory_allocated())
        times_update = (sum(times_update) - min(times_update) - max(times_update)) / (len(times_update) - 2)
        times_infer = (sum(times_infer) - min(times_infer) - max(times_infer)) / (len(times_infer) - 2)
        print('Update Time:', times_update)
        print('Infer Time:', times_infer)
        print('Latency:', times_update + times_infer)
        test_params_flop(self.model, (1, self.args.seq_len, self.args.enc_in))

    # def predict(self, path, setting, load=False):
    #     self.update_valid()
    #     res = self.online()
    #     np.save(f'./results/{self.args.dataset}_{self.args.seq_len}_{self.args.pred_len}'
    #                            f'_{self.args.model}_{setting[-1]}_pred.npy', np.vstack(res[-1]))
    #     return None, None

    def predict(self, path, setting, load=False):

        import os
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd

        if load:
            print('Loading', path)
            self.load_checkpoint(path)

        if hasattr(self, "update_valid"):
            self.update_valid()


        ret = self.online()
        mse_val = mae_val = None
        predictions = None

        if isinstance(ret, (tuple, list)):
            if len(ret) == 4:

                mse_val, mae_val, _, predictions = ret
            elif len(ret) == 3:

                mse_val, mae_val, _ = ret

        preds_seq = []
        if predictions:
            for p in predictions:
                preds_seq.append(np.asarray(p))
        if not preds_seq and getattr(self, 'pred_list', None):
            preds_seq = [np.asarray(p) for p in self.pred_list]

        if not preds_seq:
            raise RuntimeError("No predictions captured. Ensure online() fills 'predictions' or 'self.pred_list'.")

        preds_arr = np.vstack(preds_seq)

        # trues come from self.true_list
        if not getattr(self, 'true_list', None):
            raise RuntimeError("No ground truths captured. Ensure online() fills 'self.true_list'.")
        trues_arr = np.vstack([np.asarray(t) for t in self.true_list])


        def ensure_3d(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x)
            if x.ndim == 1:  #(T,)
                return x[:, None, None]
            if x.ndim == 2:  #(T, pred_len)
                return x[:, :, None]
            return x

        preds_arr = ensure_3d(preds_arr)
        trues_arr = ensure_3d(trues_arr)

        T, pred_len, C = preds_arr.shape
        assert trues_arr.shape[:2] == (T,
                                       pred_len), f"trues shape {trues_arr.shape} mismatches preds shape {preds_arr.shape}"

        features_mode = getattr(self.args, 'features')  #'M' or 'MS'


        target_idx = getattr(self.args, 'target_variate')
        print('TARGET IDX',{target_idx})
        print('Number of channels', {C})

        if features_mode == 'MS' and C != 1:
            preds_arr = preds_arr[:, :, [target_idx]]
            trues_arr = trues_arr[:, :, [target_idx]]
            C = 1

        save_dir = os.path.join('./results', setting)
        os.makedirs(save_dir, exist_ok=True)

        np.save(os.path.join(save_dir, 'preds.npy'), preds_arr)
        np.save(os.path.join(save_dir, 'trues.npy'), trues_arr)

        ms_suffix_on = bool(getattr(self.args, 'ms_channel_suffix', True))
        one_index = bool(getattr(self.args, 'channel_one_indexed', True))
        chan_label = (target_idx + 1) if one_index else target_idx

        cols = []
        data_cols = []
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

        print("Saved to:")
        print(" ", os.path.join(save_dir, 'preds.npy'))
        print(" ", os.path.join(save_dir, 'trues.npy'))
        print(" ", csv_path)

        if hasattr(self, 'save_param_df'):
            self.save_param_df(os.path.join('./results', setting), 'online_params.csv')

        return ret



class Exp_ER(Exp_Online):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer = Buffer(500, self.device)
        self.count = 0

    def train_loss(self, criterion, batch, outputs):
        loss = super().train_loss(criterion, batch, outputs)
        if not self.buffer.is_empty():
            buff = self.buffer.get_data(8)
            out = self.forward(buff[:-1])
            if isinstance(outputs, (tuple, list)):
                out = out[0]
            loss += 0.2 * criterion(out, buff[1])
        return loss

    def _update_online(self, batch, criterion, optimizer, scaler=None):
        loss, outputs = self._update(batch, criterion, optimizer, scaler=None)
        idx = self.count + torch.arange(batch[1].size(0)).to(self.device)
        self.count += batch[1].size(0)
        self.buffer.add_data(*(batch + (idx,)))
        return loss, outputs


class Exp_DERpp(Exp_ER):

    def train_loss(self, criterion, batch, outputs):
        loss = Exp_Online.train_loss(self, criterion, batch, outputs)
        if not self.buffer.is_empty():
            buff = self.buffer.get_data(8)
            out = self.forward(buff[:-1])
            if isinstance(outputs, (tuple, list)):
                out = out[0]
            loss += 0.2 * criterion(buff[-1], out)
        return loss

    def _update_online(self, batch, criterion, optimizer, scaler=None):
        loss, outputs = Exp_Online._update_online(self, batch, criterion, optimizer, scaler)
        self.count += batch[1].size(0)
        if isinstance(outputs, (tuple, list)):
            self.buffer.add_data(*(batch + [outputs[0]]))
        else:
            self.buffer.add_data(*(batch + [outputs]))
        return loss, outputs


class Exp_FSNet(Exp_Online):
    def __init__(self, args):
        super().__init__(args)

    def _update(self, *args, **kwargs):
        ret = super()._update(*args, **kwargs)
        if hasattr(self.model, 'store_grad'):
            self.model.store_grad()
        return ret

    def vali(self, *args, **kwargs):
        if not hasattr(self.model, 'try_trigger_'):
            return super().vali(*args, **kwargs)
        else:
            self.model.try_trigger_(True)
            ret = super().vali(*args, **kwargs)
            self.model.try_trigger_(False)
            return ret

    def online(self, *args, **kwargs):
        if not hasattr(self.model, 'try_trigger_'):
            return super().online(*args, **kwargs)
        else:
            self.model.try_trigger_(True)
            ret = super().online(*args, **kwargs)
            self.model.try_trigger_(False)
            return ret

    def analysis_online(self):
        if hasattr(self.model, 'try_trigger_'):
            self.model.try_trigger_(True)
        return super().analysis_online()


class Exp_OneNet(Exp_FSNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opt_w = optim.Adam([self.model.weight], lr=self.args.learning_rate_w)
        self.opt_bias = optim.Adam(self.model.decision.parameters(), lr=self.args.learning_rate_bias)
        self.bias = torch.zeros(self.args.enc_in, device=self.model.weight.device)

    def _select_optimizer(self, filter_frozen=True, return_self=True, model=None):
        if model is None or isinstance(model, OneNet):
            return super()._select_optimizer(filter_frozen, return_self, model=self.model.backbone)
        return super()._select_optimizer(filter_frozen, return_self, model=model)

    def state_dict(self, *args, **kwargs):
        destination = super().state_dict(*args, **kwargs)
        destination['opt_w'] = self.opt_w.state_dict()
        destination['opt_bias'] = self.opt_bias.state_dict()
        return destination

    # def load_state_dict(self, state_dict, model=None):
    #     self.model.bias.data = state_dict['model']['bias']
    #     return super().load_state_dict(state_dict, model)

    def _build_model(self, model=None, framework_class=None):
        if self.args.model not in ['TCN', 'FSNet', 'TCN_Ensemble', 'FSNet_Ensemble']:
            framework_class = [Model_Ensemble, OneNet]
        else:
            framework_class = OneNet
        return super()._build_model(model, framework_class=framework_class)

    def train_loss(self, criterion, batch, outputs):
        return super().train_loss(criterion, batch, outputs[1]) + super().train_loss(criterion, batch, outputs[2])

    def vali(self, vali_data, vali_loader, criterion):
        self.bias = torch.zeros(self.args.enc_in, device=self.model.weight.device)
        ret = super().vali(vali_data, vali_loader, criterion)
        self.phase = None
        return ret

    def update_valid(self, valid_data=None):
        self.bias = torch.zeros(self.args.enc_in, device=self.model.weight.device)
        return super().update_valid(valid_data)

    def forward(self, batch):
        b, t, d = batch[1].shape
        if hasattr(self, 'phase') and self.phase in self.online_phases:
            weight = self.model.weight.view(1, 1, -1).repeat(b, t, 1)
            bias = self.bias.view(-1, 1, d)
            loss1 = F.sigmoid(weight + bias.repeat(1, t, 1)).view(b, t, d)
        else:
            loss1 = F.sigmoid(self.model.weight).view(1, 1, -1)
            loss1 = loss1.repeat(b, t, 1)
        batch = batch + [loss1, 1 - loss1]
        return super().forward(batch)

    def _update(self, batch, criterion, optimizer, scaler=None):
        batch_y = batch[1]
        b, t, d = batch_y.shape

        loss, (outputs, y1, y2) = super()._update(batch, criterion, optimizer, scaler)

        loss_w = criterion(outputs, batch_y)
        loss_w.backward()
        self.opt_w.step()
        self.opt_w.zero_grad()

        y1_w, y2_w = y1.detach(), y2.detach()
        true_w = batch_y.detach()
        loss1 = F.sigmoid(self.model.weight).view(1, 1, -1)
        loss1 = loss1.repeat(b, t, 1)
        inputs_decision = torch.cat([loss1 * y1_w, (1 - loss1) * y2_w, true_w], dim=1)
        bias = self.model.decision(inputs_decision.permute(0, 2, 1)).view(b, 1, -1)
        weight = self.model.weight.view(1, 1, -1).repeat(b, t, 1)
        loss1 = F.sigmoid(weight + bias.repeat(1, t, 1))
        loss2 = 1 - loss1
        loss_bias = criterion(loss1 * y1_w + loss2 * y2_w, true_w)
        loss_bias.backward()
        self.opt_bias.step()
        self.opt_bias.zero_grad()

        return loss / 2, outputs

    def _update_online(self, batch, criterion, optimizer, scaler=None):
        batch_y = batch[1]
        b, t, d = batch_y.shape

        loss, (outputs, y1, y2) = super()._update(batch, criterion, optimizer, scaler)

        y1_w, y2_w = y1.detach(), y2.detach()
        loss1 = F.sigmoid(self.model.weight).view(1, 1, -1).repeat(b, t, 1)
        inputs_decision = torch.cat([loss1 * y1_w, (1 - loss1) * y2_w, batch_y], dim=1)
        self.bias = self.model.decision(inputs_decision.permute(0, 2, 1))
        weight = self.model.weight.view(1, 1, -1).repeat(b, t, 1)
        bias = self.bias.view(b, 1, -1)
        loss1 = F.sigmoid(weight + bias.repeat(1, t, 1))
        loss2 = 1 - loss1

        outputs_bias = loss1 * y1_w + loss2 * y2_w
        loss_bias = criterion(outputs_bias, batch_y)
        loss_bias.backward()
        self.opt_bias.step()
        self.opt_bias.zero_grad()

        loss1 = F.sigmoid(self.model.weight).view(1, 1, -1)
        loss1 = loss1.repeat(b, t, 1)
        loss_w = criterion(loss1 * y1_w + (1 - loss1) * y2_w, batch_y)
        loss_w.backward()
        self.opt_w.step()
        self.opt_w.zero_grad()
        return loss / 2, outputs

