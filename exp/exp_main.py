import importlib

from torch.utils.data import DataLoader
import shutil
import tempfile
import models.normalization
import settings
from data_provider.data_factory import get_dataset
from exp.exp_basic import Exp_Basic
from util.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop, load_model_compile
from util.metrics import metric, update_metrics, calculate_metrics

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import pandas as pd
import os
import time

import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
    
    def _unfreeze(self, model):
        pass

    @property
    def _model(self):
        if self.args.local_rank >= 0:
            return self.model.module
        return self.model

    def _build_model(self, model=None, framework_class=None):
        if model is None:
            if self.args.model.endswith('_Ensemble'):
                model = importlib.import_module(f'models.{self.args.model[:-len("_Ensemble")]}').Model_Ensemble(
                    self.args).float()
            else:
                model = importlib.import_module(f'models.{self.args.model}').Model(self.args).float()
                # if self.args.ensemble:
                #     if framework_class is not None:
                #         framework_class = [Model_Ensemble, framework_class, ]

        if self.args.normalization and self.args.online_method != 'OneNet' and self.args.model != 'FSNet_Ensemble':
            model = models.normalization.ForecastModel(model, num_features=self.args.enc_in, seq_len=self.args.seq_len,
                                                       process_method=self.args.normalization)

        if hasattr(self.args, 'load_path'):
            if not self.args.freeze:
                self.model_optim = self._select_optimizer(model=model.to(self.device))  # Otherwise, no need to reload its optimizer
            print('Load checkpoints from', self.args.load_path)
            if self.args.load_path and os.path.exists(self.args.load_path):
                model = self.load_checkpoint(self.args.load_path, model)
            else:
                print(f"[INFO] No checkpoint found at {self.args.load_path}. Starting fresh.")

            if self.model_optim is not None:
                print('Learning rate of model_optim is', self.model_optim.param_groups[0]['lr'])

            if self.args.freeze:
                model.requires_grad_(False)

        model_params = sum([param.nelement() for param in model.parameters()])
        if framework_class is not None:
            if isinstance(framework_class, list):
                for cls in framework_class:
                    model = cls(model, self.args)
            else:
                model = framework_class(model, self.args)
            new_model_params = sum([param.nelement() for param in model.parameters()])
            print(f'Number of Params: {model_params} -> {new_model_params} (+{new_model_params - model_params})')
            self.model_params = model_params
            if self.model_optim is not None:
                param_set = set()
                for group in self.model_optim.param_groups:
                    param_set.update(set(group['params']))
                new_params = list(filter(lambda p: p not in param_set and p.requires_grad, model.parameters()))
                if len(new_params) > 0:
                    self.model_optim.add_param_group({'params': new_params})

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        elif self.args.local_rank != -1:
            model = model.to(self.device)
            model = DDP(model, device_ids=[self.args.local_rank], output_device=self.args.local_rank,
                        find_unused_parameters=self.args.find_unused_parameters)
        if torch.__version__ >= '2' and self.args.compile:
            print('Compile the model by Pytorch 2.0')
            model = torch.compile(model)
        return model

    def _process_batch(self, batch):
        batch = super()._process_batch(batch)
        batch_x, batch_y = batch[:2]

        if self.args.model in settings.need_x_y_mark:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch[:4]


            batch_x_mark = batch_x_mark[:, :self.args.seq_len, :]
            batch_y_mark = batch_y_mark[:, :self.args.label_len + self.args.pred_len, :]

            if not hasattr(self, "_printed_batch_shapes"):
                self._printed_batch_shapes = True
                print("batch_x:", batch_x.shape)
                print("batch_y:", batch_y.shape)
                print("batch_x_mark:", batch_x_mark.shape)
                print("batch_y_mark:", batch_y_mark.shape)

            # decoder input for TimeXer-style models
            dec_inp = torch.zeros_like(batch_x[:, -self.args.pred_len:, :])
            dec_inp = torch.cat([batch_x[:, -self.args.label_len:, :], dec_inp], dim=1)

            inp = [batch_x, batch_x_mark, dec_inp, batch_y_mark] + list(batch[4:])

        elif self.args.model in settings.need_x_mark or (
                hasattr(self.args, 'online_method') and self.args.online_method == 'OneNet'):
            batch = batch[:3] + batch[4:]
            inp = [batch_x] + list(batch[2:])

        else:
            batch = batch[:2] + batch[4:]
            inp = [batch_x] + list(batch[2:])

        return inp

    def vali(self, vali_data, vali_loader, criterion):
        self.phase = 'val'
        total_loss = []
        self.model.eval()

        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                outputs = self.forward(batch)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                true = batch[self.label_position]
                if not self.args.pin_gpu:
                    true = true.to(self.device)
                #newly added#

                loss = criterion(outputs, true)
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        if self.args.local_rank != -1:
            total_loss = torch.tensor(total_loss, device=self.device)
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            total_loss = total_loss.item()
        return total_loss

    def train(self, setting, train_data=None, train_loader=None, vali_data=None, vali_loader=None):
        if train_data is None:
            train_data, train_loader = self._get_data(flag='train')
        if vali_data is None and not self.args.train_only:
            vali_data, vali_loader = self._get_data(flag='val')

        if self.args.checkpoints:
            path = os.path.join(self.args.checkpoints, setting)
        else:
            path = None

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        if self.args.lradj == 'TST':
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=self.args.pct_start,
                                                epochs=self.args.train_epochs,
                                                max_lr=self.args.learning_rate)
        elif self.args.model == 'GPT4TS':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.tmax, eta_min=1e-8)
        else:
            scheduler = None

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            if self.args.local_rank != -1:
                train_loader.sampler.set_epoch(epoch)
                if hasattr(self, 'online_phases') and 'val' not in self.online_phases:
                    vali_loader.sampler.set_epoch(epoch)

            self.model.train()
            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                self.phase = 'train'
                iter_count += 1
                loss, _ = self._update(batch, criterion, model_optim, scaler)
                train_loss.append(loss.item())
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            self.phase = 'train'
            train_loss = np.average(train_loss)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                epoch + 1, train_steps, train_loss), end=' ')
            if not self.args.train_only:
                if epoch >= self.args.begin_valid_epoch:
                    vali_loss = self.vali(vali_data, vali_loader, criterion)
                    # test_loss = self.vali(test_data, test_loader, criterion)
                    print("Vali Loss: {:.7f}".format(vali_loss))
                    early_stopping(vali_loss, self, path)
                else:
                    print()
            else:
                early_stopping(train_loss, self, path)
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        if self.args.train_epochs > 0:
            print('Best Valid MSE:', -early_stopping.best_score)
            self.load_state_dict(early_stopping.best_checkpoint,
                                 strict=not (hasattr(self.args, 'freeze') and self.args.freeze))

            if path and self.args.local_rank <= 0:
                if not os.path.exists(path):
                    os.makedirs(path)
                print('Save checkpoint to', path)
                torch.save(self.state_dict(local_rank=self.args.local_rank), path + '/' + 'checkpoint.pth')
        return self.model, train_data, train_loader, vali_data, vali_loader

    def test(self, setting, test_data=None, test_loader=None, test=0, target_variate=100000):
        self.phase = 'test'
        if test_data is None:
            test_data, test_loader = self._get_data(flag='test')

        if test:
            path = os.path.join("checkpoints", setting, 'checkpoint.pth')
            print('Loading', path)
            self.load_checkpoint(path)

        self.model.eval()
        statistics = {k: 0 for k in ['total', 'y_sum', 'MSE', 'MAE', "SMAPE"]}
        target_variate = getattr(self.args, 'target_variate')
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                outputs = self.forward(batch)
                true = batch[self.label_position]
                if not self.args.pin_gpu:
                    true = true.to(self.device)

                if i == 0:
                    print("TEST before metrics | outputs:", outputs.shape, "true:", true.shape)


                if true.dim() == 3 and true.size(-1) > 1 and outputs.dim() == 3 and outputs.size(-1) == 1:

                    tv = getattr(self.args, "target_variate", None)
                    if tv is not None and tv < true.size(-1):
                        true = true[:, :, tv:tv + 1]
                    else:
                        true = true[:, :, -1:]

                update_metrics(outputs, true, statistics, target_variate)

        metrics = calculate_metrics(statistics)
        print('target variate is', target_variate)
        mse, mae , rmse, smape = metrics['MSE'], metrics['MAE'], metrics['RMSE'], metrics["SMAPE"]
        print("running test")
        print('rmse:{}, mae:{}, smape:{}'.format(rmse, mae, smape))
        return rmse, mae, smape, test_data, test_loader

    def predict(self, path, setting, load=False):
        import os
        import numpy as np
        import pandas as pd
        from util.metrics import update_metrics, calculate_metrics

        if load:
            print('Loading', path)
            self.load_checkpoint(path)

        self.phase = 'predict'
        self.model.eval()


        test_data, test_loader = self._get_data(flag='test')

        preds, trues = [], []

        statistics = {k: 0 for k in ['total', 'y_sum', 'MSE', 'MAE', 'SMAPE']}
        target_variate = getattr(self.args, 'target_variate', None)

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                outputs = self.forward(batch)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]

                true = batch[self.label_position]
                if not self.args.pin_gpu:
                    true = true.to(self.device)


                update_metrics(outputs, true, statistics, target_variate)


                preds.append(outputs.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())

        metrics = calculate_metrics(statistics)
        rmse = metrics['RMSE']
        mae = metrics['MAE']
        smape = metrics['SMAPE']

        print('running predict (test-equivalent)')
        print('target variate is', target_variate)
        print(f'rmse:{rmse}, mae:{mae}, smape:{smape}')

        preds = np.vstack(preds)
        trues = np.vstack(trues)

        if preds.ndim == 2:
            preds = preds[..., None]
        if trues.ndim == 2:
            trues = trues[..., None]

        N, H, C = preds.shape
        features_mode = getattr(self.args, 'features', 'M')


        if target_variate is None:
            target_idx = 0
        else:
            target_idx = int(target_variate)

        if features_mode == 'MS' and C == 1:
            target_idx = 0

        if target_idx < 0:
            target_idx = 0
        if target_idx >= C:
            target_idx = C - 1

        preds_c = preds[..., target_idx]
        trues_c = trues[..., target_idx]


        out_dir = os.path.join('./results', setting)
        os.makedirs(out_dir, exist_ok=True)

        np.save(os.path.join(out_dir, 'preds.npy'), preds)
        np.save(os.path.join(out_dir, 'trues.npy'), trues)

        one_index = True
        chan_label = (target_idx + 1) if one_index else target_idx
        ms_suffix = f"_channel_{chan_label}" if features_mode == 'MS' else ""
        suffix = ms_suffix if features_mode == 'MS' else ""

        pred_cols = [f'pred_h{h + 1}{suffix}' for h in range(H)]
        true_cols = [f'true_h{h + 1}{suffix}' for h in range(H)]
        preds_flat = preds_c.reshape(N, H)
        trues_flat = trues_c.reshape(N, H)

        df = pd.DataFrame(np.hstack([preds_flat, trues_flat]), columns=pred_cols + true_cols)
        csv_path = os.path.join(out_dir, 'preds_trues_wide.csv')
        df.to_csv(csv_path, index=False)

        print("Saved:")
        print(" ", csv_path)
        print(" ", os.path.join(out_dir, 'preds.npy'))
        print(" ", os.path.join(out_dir, 'trues.npy'))

        return rmse, mae, smape



    def analysis(self):
        data = get_dataset(self.args, 'test', self.device, wrap_class=self.args.wrap_data_class,
                                  **self.wrap_data_kwargs)
        times_infer = []
        print('GPU Mem:', torch.cuda.max_memory_allocated())
        self.model.eval()
        with torch.no_grad():
            for i in range(50):
                start_time = time.time()
                current_data = [d.unsqueeze(0).to(self.device) for d in data[i]]
                self.forward(current_data)
                if i > 10:
                    times_infer.append(time.time() - start_time)
                if i == 30:
                    break
        print('Final GPU Mem:', torch.cuda.max_memory_allocated())
        times_infer = (sum(times_infer) - min(times_infer) - max(times_infer)) / (len(times_infer) - 2)
        print('Latency:', times_infer)
        test_params_flop(self.model, (1, self.args.seq_len, self.args.enc_in))
