import copy
import numpy as np
import torch
from tqdm import tqdm
import os
from adapter import proceed


from data_provider.data_factory import get_dataset, get_dataloader
from data_provider.data_loader import Dataset_Recent
from exp import Exp_Online


class Exp_Proceed(Exp_Online):
    def __init__(self, args):
        args = copy.deepcopy(args)
        args.merge_weights = 1
        super(Exp_Proceed, self).__init__(args)
        self.online_phases = ['val', 'test', 'online']
        self.mean_dim = args.seq_len

    def online(self, online_data=None, target_variate=10000, phase='test', show_progress=False):
        if phase == 'val' and self.args.val_online_lr:
            lr = self.model_optim.param_groups[0]['lr']
            for j in range(len(self.model_optim.param_groups)):
                self.model_optim.param_groups[j]['lr'] = self.args.online_learning_rate
        self.model_optim.zero_grad()
        self._model.freeze_adapter(True)
        ret = super().online(online_data, target_variate, phase, show_progress)
        self._model.freeze_adapter(False)
        if phase == 'val' and self.args.val_online_lr:
            for j in range(len(self.model_optim.param_groups)):
                self.model_optim.param_groups[j]['lr'] = lr
        return ret

    def update_valid(self, valid_data=None, valid_dataloader=None):
        self.phase = 'online'
        if valid_data is None:
            valid_data = get_dataset(self.args, 'val', self.device,
                                     wrap_class=self.args.wrap_data_class + [Dataset_Recent],
                                     **self.wrap_data_kwargs, take_post=self.args.pred_len - 1)
        valid_loader = get_dataloader(valid_data, self.args, 'online')
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None
        self.model.train()
        predictions = []
        if not self.args.joint_update_valid:
            for i, (recent_batch, current_batch) in enumerate(tqdm(valid_loader, mininterval=10)):
                self._model.freeze_bias(False)
                self._model.freeze_adapter(True)
                self._update_online(recent_batch, criterion, model_optim, scaler, flag_current=False)
                self._model.freeze_bias(True)
                if not self.args.freeze:
                    self._model.backbone.requires_grad_(False)
                self._model.freeze_adapter(False)
                _, outputs = self._update_online(current_batch, criterion, model_optim, scaler, flag_current=True)
                if not self.args.freeze:
                    self._model.backbone.requires_grad_(True)
                if self.args.do_predict:
                    if isinstance(outputs, (tuple, list)):
                        outputs = outputs[0]
                    predictions.append(outputs.detach().cpu().numpy())
            self._model.freeze_bias(False)
        else:
            for i, (recent_batch, current_batch) in enumerate(tqdm(valid_loader, mininterval=10)):
                self._update_online(recent_batch, criterion, model_optim, scaler, flag_current=True)
                if self.args.do_predict:
                    self.model.eval()
                    with torch.no_grad():
                        outputs = self.forward(current_batch)
                    if isinstance(outputs, (tuple, list)):
                        outputs = outputs[0]
                    predictions.append(outputs.detach().cpu().numpy())
                    self.model.train()
        self.model_optim.zero_grad()
        self._model.freeze_adapter(True)
        trainable_params = sum([param.nelement() if param.requires_grad else 0 for param in self._model.parameters()])
        print(f'Trainable Params: {trainable_params}', '({:.1f}%)'.format(trainable_params / self.model_params * 100))
        return predictions

    def _build_model(self, model=None, framework_class=None):
        model = super()._build_model(model, framework_class= proceed.Proceed)
        print(model)
        return model

    def _update(self, batch, criterion, optimizer, scaler=None):
        self._model.flag_update = True
        loss, outputs = super()._update(batch, criterion, optimizer, scaler)
        self._model.recent_batch = torch.cat([batch[0], batch[1]], -2)
        self._model.flag_update = False
        return loss, outputs

    def _update_online(self, batch, criterion, optimizer, scaler=None, flag_current=False):
        self._model.flag_online_learning = True
        self._model.flag_current = flag_current
        loss, outputs = super()._update_online(batch, criterion, optimizer, scaler)
        self._model.recent_batch = torch.cat([batch[0], batch[1]], -2)
        self._model.flag_online_learning = False
        self._model.flag_current = not flag_current
        return loss, outputs

    def analysis_online(self):
        self._model.freeze_adapter(True)
        return super().analysis_online()

    def predict(self, path, setting, load=False):
        import os
        import numpy as np
        import pandas as pd

        if load:
            print("Loading", path)
            self.load_checkpoint(path)


        test_data, test_loader = self._get_data(flag='test')

        target_variate = getattr(self.args, "target_variate", None)


        ret = self.online(
            online_data=test_data,
            target_variate=target_variate,
            phase='test',
            show_progress=True
        )

        preds_seq = []
        if isinstance(ret, (tuple, list)) and len(ret) >= 4:
            predictions = ret[3]
            if predictions:
                preds_seq = [np.asarray(p) for p in predictions]

        if not preds_seq and getattr(self, "pred_list", None):
            preds_seq = [np.asarray(p) for p in self.pred_list]

        if not preds_seq:
            raise RuntimeError(
                "No predictions captured. Ensure Exp_Online.online() populates predictions/self.pred_list.")

        if not getattr(self, "true_list", None):
            raise RuntimeError("No ground truths captured. Ensure Exp_Online.online() populates self.true_list.")

        preds_arr = np.vstack(preds_seq)
        trues_arr = np.vstack([np.asarray(t) for t in self.true_list])

        def ensure_3d(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x)
            if x.ndim == 1:  #(T,)
                return x[:, None, None]
            if x.ndim == 2:  #(T, H)
                return x[:, :, None]
            return x

        preds_arr = ensure_3d(preds_arr)
        trues_arr = ensure_3d(trues_arr)

        T = min(preds_arr.shape[0], trues_arr.shape[0])
        preds_arr = preds_arr[:T]
        trues_arr = trues_arr[:T]

        T, H, C = preds_arr.shape


        features_mode = getattr(self.args, "features", "M")
        target_idx = getattr(self.args, "target_variate", None)


        if target_idx is None:
            target_idx = 0
        target_idx = int(target_idx)
        target_idx = max(0, min(target_idx, C - 1))

        if features_mode == "MS" and C != 1:
            preds_arr = preds_arr[:, :, [target_idx]]
            trues_arr = trues_arr[:, :, [target_idx]]
            C = 1

        if trues_arr.shape[-1] != C:
            if C == 1 and trues_arr.shape[-1] > 1:
                trues_arr = trues_arr[:, :, [target_idx]]
            else:
                raise RuntimeError(f"Cannot align channel dims: preds C={C}, trues C={trues_arr.shape[-1]}")

        save_dir = os.path.join("./results", setting)
        os.makedirs(save_dir, exist_ok=True)

        np.save(os.path.join(save_dir, "preds.npy"), preds_arr)
        np.save(os.path.join(save_dir, "trues.npy"), trues_arr)


        ms_suffix_on = bool(getattr(self.args, "ms_channel_suffix", True))
        one_index = bool(getattr(self.args, "channel_one_indexed", True))
        chan_label = (target_idx + 1) if one_index else target_idx
        single_suffix = f"_channel_{chan_label}" if (features_mode == "MS" and ms_suffix_on) else ""

        cols, data_cols = [], []
        if C == 1:
            p = preds_arr[:, :, 0]
            y = trues_arr[:, :, 0]
            for h in range(H):
                cols.append(f"pred_h{h + 1}{single_suffix}")
                data_cols.append(p[:, h])
            for h in range(H):
                cols.append(f"true_h{h + 1}{single_suffix}")
                data_cols.append(y[:, h])
        else:
            for c in range(C):
                for h in range(H):
                    cols.append(f"pred_h{h + 1}_c{c}")
                    data_cols.append(preds_arr[:, h, c])
                for h in range(H):
                    cols.append(f"true_h{h + 1}_c{c}")
                    data_cols.append(trues_arr[:, h, c])

        df = pd.DataFrame(np.column_stack(data_cols), columns=cols)
        csv_path = os.path.join(save_dir, "preds_trues_wide.csv")
        df.to_csv(csv_path, index=False)

        print("Saved to:")
        print(" ", os.path.join(save_dir, "preds.npy"))
        print(" ", os.path.join(save_dir, "trues.npy"))
        print(" ", csv_path)

        return ret
