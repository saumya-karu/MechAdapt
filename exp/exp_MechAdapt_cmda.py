# exp/exp_MechAdapt_cmda.py

import copy
import os
from typing import Optional, List, Union
import torch
import torch.optim as optim
from exp import Exp_Proceed
from util.neural_lagged_graph import NeuralLaggedCausalGraph
from util.graph_drift_gate import GraphDriftGate
from util.mechanism_adapter import MechanismAdapter
from util.pcsd_gate_cmda import PCSDGateCMDA
from util.light_adapter import LightAdapter
from adapter.MechAdapt_cmda import MechAdapt
from util.covariate_recalib import CovariateRecalibration
BatchType = Union[tuple, list]


class Exp_MechAdapt(Exp_Proceed):

    def __init__(self, args):
        args = copy.deepcopy(args)


        args.merge_weights = 1

        super().__init__(args)
        from adapter.MechAdapt_cmda import MechAdapt
        from adapter.proceed import Proceed

        base = self._model

        if isinstance(base, Proceed):
            backbone = base.backbone
            wrapped = MechAdapt(backbone, self.args).to(self.device)


            if hasattr(self.model, "module"):
                self.model.module = wrapped
            else:
                self.model = wrapped


        assert hasattr(self._model, "set_drift"), f"MechAdapt not active, got {type(self._model)}"
        assert hasattr(self._model, "encode_graph_delta"), "encode_graph_delta missing"

        if hasattr(self._model, "generator"):
            gen_params = [p for p in self._model.generator.parameters() if p.requires_grad]


            opt_params = [p for g in self.model_optim.param_groups for p in g["params"]]
            opt_ids = {id(p) for p in opt_params}

            missing = [p for p in gen_params if id(p) not in opt_ids]
            if len(missing) > 0:
                self.model_optim.add_param_group({"params": missing})
                print(f"[FIX] Added {len(missing)} generator params to model_optim", flush=True)


        self.device = getattr(
            self, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.pred_len = int(getattr(self.args, "pred_len", 24))
        self.seq_len = int(getattr(self.args, "seq_len", 96))
        self.dataset= str(getattr(self.args, "dataset"))
        self.online_method = str(getattr(self.args, "online_method"))
        self.model_name = str(getattr(self.args, "model"))
        self.target_variate = getattr(self.args, "target_variate", None)
        self.target_variable = str(
            getattr(self.args, "target_variable", getattr(self.args, "target", "Y"))
        )

        self.causal_channel_indices: Optional[List[int]] = getattr(
            self.args, "causal_channel_indices", None
        )
        self.y_index_in_causal = int(getattr(self.args, "y_index_in_causal", -1))


        cra_channels = int(
            getattr(self.args, "enc_in", 0) or getattr(self.args, "dec_in", 0) or 4
        )

        self.cra = CovariateRecalibration(
            cra_channels,
            learn_affine=bool(getattr(self.args, "cra_learn_affine", True)),
            clip_gamma=float(getattr(self.args, "cra_clip_gamma", 3.0)),
            ref_ema=float(getattr(self.args, "cra_ref_ema", 0.999)),
            warmup_steps=int(getattr(self.args, "cra_warmup", 50)),
            device=self.device,
        )

        self.cra_lr = float(getattr(self.args, "cra_lr", 1e-3))
        self.cra_steps = int(getattr(self.args, "cra_steps", 1))
        self.cra_optim = None
        if any(p.requires_grad for p in self.cra.parameters()):
            self.cra_optim = optim.Adam(
                [p for p in self.cra.parameters() if p.requires_grad], lr=self.cra_lr
            )


        self.cra_train_ws = float(getattr(self.args, "cra_train_ws", 0.6))  # train if w_s > this
        self.cra_block_wm = float(getattr(self.args, "cra_block_wm", 0.2))  # and w_m < this

        graph_lags = int(getattr(self.args, "cmda_tau_max", 24))
        graph_window = int(getattr(self.args, "cmda_graph_window", 512))
        self.graph_train_every = int(getattr(self.args, "cmda_graph_train_every", 10))
        self.graph_train_steps = int(getattr(self.args, "cmda_graph_train_steps", 20))
        self.graph_lr = float(getattr(self.args, "cmda_graph_lr", 1e-3))

        self.causal_dim = int(getattr(self.args, "cmda_causal_dim", 0))  # if 0 infer from seq_x

        self.graph_estimator = NeuralLaggedCausalGraph(
            tau_max=graph_lags,
            window=graph_window,
            y_index=self.y_index_in_causal,
            device=self.device,
        )


        self.graph_gate = GraphDriftGate(
            threshold=float(getattr(self.args, "cmda_graph_thresh", 0.01)),
            ema=float(getattr(self.args, "cmda_graph_ema", 0.98)),
            warmup=int(getattr(self.args, "cmda_graph_warmup", 50)),
            stable_thresh=float(getattr(self.args, "cmda_graph_stable_thresh", 0.01)),
            stable_patience=int(getattr(self.args, "cmda_graph_stable_patience", 40)),


            y_index=int(getattr(self.args, "y_index_in_causal", -1)),  # usually -1
            beta_l2=float(getattr(self.args, "cmda_graph_beta_l2", 1)),
            normalize=str(getattr(self.args, "cmda_graph_norm", "global")),  # row recommended
            freeze_ref_during_warmup=True,
            update_ref_when_stable=True,
        )


        self.pcsd = PCSDGateCMDA(
            model=self._model,
            include_keywords=("generator", "mlp1", "mlp2"),
            feat_mode=str(getattr(self.args, "pcsd_feat_mode", "l2_l1")),
            ema=float(getattr(self.args, "pcsd_ema", 0.95)),
            threshold=float(getattr(self.args, "pcsd_thresh", 0.15)),
            warmup_steps=int(getattr(self.args, "pcsd_warmup", 10)),
            freeze_ref_on_fire=bool(getattr(self.args, "pcsd_freeze_ref_on_fire", True)),
            device=self.device,
        )

        adapter_hidden = int(getattr(self.args, "cmda_adapter_hidden", 64))
        adapter_dropout = float(getattr(self.args, "cmda_adapter_dropout", 0.0))
        adapter_use_window = bool(getattr(self.args, "cmda_adapter_use_window", True))
        adapter_window = int(getattr(self.args, "cmda_adapter_window", min(64, self.seq_len)))

        if self.causal_channel_indices is not None:
            causal_dim = len(self.causal_channel_indices)
        else:
            causal_dim = int(getattr(self.args, "enc_in", 0) or getattr(self.args, "dec_in", 0) or 4)

        self.mechanism_adapter = MechanismAdapter(
            input_dim=causal_dim,
            pred_len=self.pred_len,
            hidden=adapter_hidden,
            dropout=adapter_dropout,
            use_window=adapter_use_window,
            window=adapter_window,
        ).to(self.device)


        self.mechanism_adapter.build()

        self.adapter_lr = float(getattr(self.args, "cmda_adapter_lr", 1e-4))
        self.adapter_steps = int(getattr(self.args, "cmda_adapter_steps", 1))
        self.adapter_optim = optim.Adam(self.mechanism_adapter.parameters(), lr=self.adapter_lr)


        self.enable_light = bool(getattr(self.args, "cmda_enable_light", True))

        light_hidden = int(getattr(self.args, "cmda_light_hidden", 128))
        light_dropout = float(getattr(self.args, "cmda_light_dropout", 0.005))
        light_window = int(getattr(self.args, "cmda_light_window", self.seq_len))

        self.light_adapter = LightAdapter(
            input_dim=None,
            pred_len=self.pred_len,
            hidden=light_hidden,
            dropout=light_dropout,
            window=light_window,
        ).to(self.device)

        self.light_lr = float(getattr(self.args, "cmda_light_lr", 1e-5))
        self.light_steps = int(getattr(self.args, "cmda_light_steps", 1))
        self.light_optim = None


        #regime logs
        self.regime_list = []


        self.causal_var_names = list(getattr(self.args, "causal_var_names", []))
        if len(self.causal_var_names) == 0:
            inferred = None
            data_path = getattr(self.args, "data_path", None)
            root_path = getattr(self.args, "root_path", None)
            if data_path is not None:
                try:
                    import pandas as pd
                    csv_path = os.path.join(str(root_path), str(data_path)) if root_path is not None else str(data_path)
                    if os.path.exists(csv_path):
                        cols = list(pd.read_csv(csv_path, nrows=1).columns)
                        if "date" in cols:
                            cols.remove("date")
                        else:
                            cols = cols[1:]
                        inferred = cols
                except Exception:
                    inferred = None

            if inferred is None:
                C = len(self.causal_channel_indices) if self.causal_channel_indices is not None else causal_dim
                inferred = [f"X{i+1}" for i in range(max(C - 1, 1))] + ["Y"]

            if self.causal_channel_indices is not None and len(inferred) >= (max(self.causal_channel_indices) + 1):
                inferred = [inferred[i] for i in self.causal_channel_indices]

            self.causal_var_names = inferred


        self.mix_alpha_graph = float(getattr(self.args, "mix_alpha_graph", 8.0))
        self.mix_alpha_pcsd = float(getattr(self.args, "mix_alpha_pcsd", 8.0))


        self.cmda_step = 0
        self.graph_log_every = int(getattr(self.args, "cmda_graph_log_every", 1000))
        self.graph_log_root = str(getattr(self.args, "cmda_graph_log_root", os.path.join("results", "graphs")))


        self.s_graph_list = []
        self.s_pcsd_list = []
        self.do_update_list = []
        self.do_graph_list = []
        self.do_pcsd_list = []
        self.w_s_list = []
        self.w_m_list = []

        self._tg_ema = None
        self._tp_ema = None
        self.thresh_ema = float(getattr(self.args, "cmda_thresh_ema", 0.9))


        self.auto_calib = bool(getattr(self.args, "cmda_auto_calib", True))
        self.calib_frac = float(getattr(self.args, "cmda_calib_frac", 0.2))
        self.calib_done = False

        self.q_graph_stable = float(getattr(self.args, "cmda_q_graph_stable", 40.0))
        self.q_graph_heavy = float(getattr(self.args, "cmda_q_graph_heavy", 50.0))
        self.q_pcsd_heavy = float(getattr(self.args, "cmda_q_pcsd_heavy", 40.0))

        self._calib_s_graph = []
        self._calib_s_pcsd = []

        from collections import deque

        self.dyn_thresh = bool(getattr(self.args, "cmda_dyn_thresh", True))
        self.dyn_window = int(getattr(self.args, "cmda_dyn_window", 300))  # W
        self.recalc_every = int(getattr(self.args, "cmda_recalc_every", 20))  # 1 = every step

        self._sg_buf = deque(maxlen=self.dyn_window)
        self._sp_buf = deque(maxlen=self.dyn_window)

        self.regime_log_path = self._reset_regime_log()

        self.pt_log_every = int(getattr(self.args, "pt_log_every", 1))  # log every step by default
        self.pt_log_root = str(getattr(self.args, "pt_log_root", os.path.join("results", "pred_true")))
        os.makedirs(self.pt_log_root, exist_ok=True)
        self.pt_log_path = os.path.join(
            self.pt_log_root, f"{self.args.dataset}_{self.online_method}_pred{self.pred_len}_predtrue.csv"
        )


        if bool(getattr(self.args, "wipe_pred_true", True)):
            try:
                if os.path.exists(self.pt_log_path):
                    os.remove(self.pt_log_path)
            except Exception as e:
                print(f"[WARN] Could not delete {self.pt_log_path}: {e}")


        self.lg_ema = None
        self.lp_ema = None

        self.sg_ema = None
        self.sp_ema = None
        self.mix_ema = float(getattr(self.args, "cmda_mix_ema", 0.90))

        self.last_sp = None
        self._allow_backbone_update = False

        self.heavy_min_steps = int(getattr(self.args, "cmda_heavy_min_steps", 20))
        self.heavy_cooldown = int(getattr(self.args, "cmda_heavy_cooldown", 10))
        self.heavy_exit_margin = float(getattr(self.args, "cmda_heavy_exit_margin", 0.02))

        self._heavy_hold = 0
        self._cooldown = 0

        self.enable_cra = bool(getattr(self.args, "cmda_enable_cra", False))

        self.heavy_out_ema = None
        self.heavy_out_ema_beta = float(getattr(self.args, "cmda_heavy_out_ema_beta", 0.85))

        self._last_regime = "none"
        self._last_w_m = 0.0
        self.recent_update_wm_thr = float(getattr(self.args, "cmda_recent_update_wm_thr", 0.6))

        self.A_ema = None
        self.A_ema_beta = float(getattr(self.args, "cmda_A_ema_beta", 0.95))
        self.dA_ema = None
        self.dA_ema_beta = float(getattr(self.args, "cmda_dA_ema_beta", 0.95))

        self.light_scale_ema = None
        self.light_scale_beta = float(getattr(self.args, "cmda_light_scale_beta", 0.9))
        self.light_delta_ema = None
        self.light_delta_beta = float(getattr(self.args, "cmda_light_delta_beta", 0.90))


        self._cmda_last_A_t = None
        self._cmda_last_A_ref = None
        self._cmda_last_drift = None
        self._cmda_last_have_graphs = False

        self._light_prev_delta_train = None
        if self.dataset in {"Weather"} and self.model_name == "iTransformer": #24 and 48
            self.q_graph_stable = 50
            self.q_graph_heavy = 60
            self.A_ema_beta =0.85 #CHECK THIS



    def _set_cmda_train_targets(self, heavy: bool):

        if hasattr(self._model, "backbone"):
            self._model.backbone.requires_grad_(False)

        if hasattr(self._model, "freeze_adapter"):
            self._model.freeze_adapter(not heavy)


        if hasattr(self._model, "freeze_bias"):
            self._model.freeze_bias(not heavy)

    @torch.no_grad()
    def _update_light_scale_from_recent(self, base: torch.Tensor, batch_recent, delta_raw: torch.Tensor):
        eps = 1e-6
        rho = float(getattr(self.args, "cmda_light_rho", 0.3))
        scale_max = float(getattr(self.args, "cmda_light_scale_max", 2.0))

        y_true = self._get_aligned_y_true(batch_recent, base)


        if base.size(-1) == 1:
            pred_target = base
        else:
            tv = self.target_variate
            if tv is None:
                tv = base.size(-1) - 1
            tv = int(max(0, min(tv, base.size(-1) - 1)))
            pred_target = base[:, :, tv:tv + 1]

        err = (pred_target - y_true)
        err_rms = err.pow(2).mean().sqrt().clamp_min(eps)
        raw_rms = delta_raw.pow(2).mean().sqrt().clamp_min(eps)

        scale = (rho * err_rms / raw_rms).clamp(0.0, scale_max)

        #EMA
        b = float(getattr(self, "light_scale_beta", 0.9))
        if self.light_scale_ema is None:
            self.light_scale_ema = float(scale.item())
        else:
            self.light_scale_ema = b * float(self.light_scale_ema) + (1.0 - b) * float(scale.item())

    def _cra_select_x(self, seq_x: torch.Tensor) -> torch.Tensor:

        y_idx = int(getattr(self.args, "y_index_in_causal", -1))
        C = seq_x.size(-1)
        if y_idx < 0:
            y_idx = C - 1
        y_idx = max(0, min(y_idx, C - 1))


        idx = [j for j in range(C) if j != y_idx]
        if len(idx) == 0:
            return seq_x
        idx = torch.as_tensor(idx, device=seq_x.device, dtype=torch.long)
        return seq_x.index_select(dim=2, index=idx)

    @torch.no_grad()
    def _cra_update_ref_from_recent(self, batch_recent):

        if self.cra is None:
            return

        seq_x = batch_recent[0]
        if not getattr(self.args, "pin_gpu", False):
            seq_x = seq_x.to(self.device)

        C = seq_x.size(-1)
        idx_x, y_idx = self._x_idx(C, seq_x.device)


        x = seq_x.index_select(2, idx_x)  #[B,L,C-1]
        mu = x.mean(dim=(0, 1))  #[C-1]
        std = x.std(dim=(0, 1), unbiased=False).clamp_min(self.cra.eps)


        if (self.cra.mu_ref is None) or (self.cra.std_ref is None) or (self.cra.mu_ref.numel() != C):
            self.cra.mu_ref = torch.zeros(C, device=seq_x.device, dtype=seq_x.dtype)
            self.cra.std_ref = torch.ones(C, device=seq_x.device, dtype=seq_x.dtype)


        self.cra.mu_ref.index_copy_(0, idx_x, mu)
        self.cra.std_ref.index_copy_(0, idx_x, std)


        self.cra.steps += 1
        if hasattr(self.cra, "initialized"):
            self.cra.initialized.fill_(True)

    @torch.no_grad()
    def _recent_forward_only(self, recent_data):

        was_training = self._model.training
        self._model.eval()


        _ = super().forward(recent_data)


        if hasattr(self._model, "recent_batch"):
            bx, by = recent_data[0], recent_data[1]
            if not getattr(self.args, "pin_gpu", False):
                bx = bx.to(self.device)
                by = by.to(self.device)
            self._model.recent_batch = torch.cat([bx, by], dim=1).detach()


        self._detach_model_state()
        self._model.train(was_training)

    def _cra_fit_on_current(self, batch_current):

        if self.cra_optim is None:
            return


        trainable = [p for p in self.cra.parameters() if p.requires_grad]
        if len(trainable) == 0:
            return

        if int(self.cra.steps.item()) < int(self.cra.warmup_steps):
            return
        if (self.cra.mu_ref is None) or (self.cra.std_ref is None):
            return
        if not bool(self.cra.initialized.item()):
            return

        seq_x = batch_current[0]
        if not isinstance(seq_x, torch.Tensor) or seq_x.dim() != 3:
            return
        if not getattr(self.args, "pin_gpu", False):
            seq_x = seq_x.to(self.device)

        C = seq_x.size(-1)
        y_idx = int(getattr(self.args, "y_index_in_causal", -1))
        if y_idx < 0:
            y_idx = C - 1
        y_idx = max(0, min(y_idx, C - 1))

        idx = [j for j in range(C) if j != y_idx]
        if len(idx) == 0:
            return
        idx_t = torch.as_tensor(idx, device=seq_x.device, dtype=torch.long)


        mu_ref = self.cra.mu_ref.detach().index_select(0, idx_t)
        std_ref = self.cra.std_ref.detach().index_select(0, idx_t)

        was_training = self.cra.training
        self.cra.train()


        with torch.set_grad_enabled(True):
            for _ in range(int(getattr(self, "cra_steps", 1))):
                self.cra_optim.zero_grad(set_to_none=True)

                #forward
                seq_hat = self.cra(seq_x)  # [B,L,C]

                if (not isinstance(seq_hat, torch.Tensor)) or (not seq_hat.requires_grad):
                    # if getattr(self, "cmda_step", 0) % 200 == 0:
                    #     print(
                    #         "[CRA dbg] seq_hat.requires_grad=False -> CRA fit skipped (detached forward or non-learnable).")
                    break


                x_hat = seq_hat.index_select(dim=2, index=idx_t)  # [B,L,C-1]
                mu_hat = x_hat.mean(dim=(0, 1))
                std_hat = x_hat.std(dim=(0, 1), unbiased=False).clamp_min(float(getattr(self.cra, "eps", 1e-5)))

                loss = (mu_hat - mu_ref).pow(2).mean() + (std_hat - std_ref).pow(2).mean()

                # final guard (never crash)
                if not loss.requires_grad:
                    # if getattr(self, "cmda_step", 0) % 200 == 0:
                    #     print("[CRA dbg] loss.requires_grad=False -> CRA fit skipped.")
                    break

                loss.backward()
                self.cra_optim.step()

        self.cra.train(was_training)
        self.cra.eval()

    def _cra_apply_to_batch_current(self, batch_current, allow_update_ref: bool = False):
        seq_x = batch_current[0]
        if not isinstance(seq_x, torch.Tensor) or seq_x.dim() != 3:
            return batch_current

        if not getattr(self.args, "pin_gpu", False):
            seq_x = seq_x.to(self.device)

        C = seq_x.size(-1)
        y_idx = int(getattr(self.args, "y_index_in_causal", -1))
        if y_idx < 0:
            y_idx = C - 1
        y_idx = max(0, min(y_idx, C - 1))

        with torch.no_grad():
            if allow_update_ref:
                self.cra.update_reference(seq_x)  # full-shape reference (safe)
            if self.cra.steps < self.cra.warmup_steps:
                return batch_current

            seq_hat = self.cra(seq_x)  #[B,L,C]


            seq_hat[:, :, y_idx:y_idx + 1] = seq_x[:, :, y_idx:y_idx + 1]

        return self._replace_seq_x(batch_current, seq_hat)

    def _init_pred_true_logger(self, H: int):
        import os, csv

        os.makedirs(os.path.dirname(self.pt_log_path), exist_ok=True)

        if getattr(self, "_pt_f", None) is not None:
            return

        write_header = not os.path.exists(self.pt_log_path)
        self._pt_f = open(self.pt_log_path, "a", newline="")  # keep handle open
        self._pt_writer = csv.writer(self._pt_f)

        if write_header:
            header = ["step", "regime", "s_graph", "s_pcsd", "w_s", "w_m"]
            header += [f"pred_h{i + 1}" for i in range(H)]
            header += [f"true_h{i + 1}" for i in range(H)]
            self._pt_writer.writerow(header)
            self._pt_f.flush()

    def _close_pred_true_logger(self):
        f = getattr(self, "_pt_f", None)
        if f is not None:
            try:
                f.flush()
            except Exception:
                pass
            try:
                f.close()
            except Exception:
                pass
        self._pt_f = None
        self._pt_writer = None

    def _log_pred_true_row(
            self,
            step: int,
            outputs: torch.Tensor,
            y_true: torch.Tensor,
            regime: str,
            sg: float,
            sp: float,
            w_s: float,
            w_m: float
    ):


        if outputs is None or y_true is None:
            return
        if not (isinstance(outputs, torch.Tensor) and isinstance(y_true, torch.Tensor)):
            return
        if outputs.dim() != 3 or y_true.dim() != 3:
            return


        H = min(outputs.size(1), y_true.size(1))
        outputs = outputs[:, :H, :]
        y_true = y_true[:, :H, :]


        tv = self.target_variate
        if tv is None:
            tv = outputs.size(-1) - 1
        tv = int(max(0, min(tv, outputs.size(-1) - 1)))

        outputs = outputs[:, :, tv:tv + 1]
        if y_true.size(-1) >= tv + 1:
            y_true = y_true[:, :, tv:tv + 1]
        else:
            y_true = y_true[:, :, -1:]

        p = outputs.detach().float().mean(dim=0).cpu().numpy().reshape(H, -1)[:, 0]
        y = y_true.detach().float().mean(dim=0).cpu().numpy().reshape(H, -1)[:, 0]


        self._init_pred_true_logger(H)

        row = [int(step), str(regime), float(sg), float(sp), float(w_s), float(w_m)]
        row += [float(v) for v in p.tolist()]
        row += [float(v) for v in y.tolist()]
        self._pt_writer.writerow(row)


        flush_every = int(getattr(self.args, "pt_flush_every", 50))
        if flush_every > 0 and (int(step) % flush_every == 0):
            self._pt_f.flush()

    def _update_dynamic_thresholds(self):
        if len(self._sg_buf) < max(200, int(0.1 * self.dyn_window)):
            return

        sg = torch.tensor(list(self._sg_buf), dtype=torch.float32, device="cpu")
        sp = torch.tensor(list(self._sp_buf), dtype=torch.float32, device="cpu")

        stable = float(torch.quantile(sg, self.q_graph_stable / 100.0).item())
        heavy = float(torch.quantile(sg, self.q_graph_heavy / 100.0).item())
        if heavy <= stable:
            heavy = stable + 1e-6

        pcsd_thr = float(torch.quantile(sp, self.q_pcsd_heavy / 100.0).item())

        b = float(getattr(self, "thresh_ema", 0.95))


        if getattr(self, "_ts_ema", None) is None:
            self._ts_ema = stable  #stable-thresh EMA
            self._tg_ema = heavy  #heavy-thresh EMA
            self._tp_ema = pcsd_thr  #pcsd-thresh EMA
        else:
            self._ts_ema = b * self._ts_ema + (1.0 - b) * stable
            self._tg_ema = b * self._tg_ema + (1.0 - b) * heavy
            self._tp_ema = b * self._tp_ema + (1.0 - b) * pcsd_thr


        self.graph_gate.stable_thresh = float(self._ts_ema)
        self.graph_gate.threshold = float(self._tg_ema)
        self.pcsd.threshold = float(self._tp_ema)


    def _finalize_calibration(self):
        sg = torch.tensor(self._calib_s_graph, dtype=torch.float32)
        sp = torch.tensor(self._calib_s_pcsd, dtype=torch.float32)

        def pct(x, q):
            return float(torch.quantile(x, q / 100.0).item())

        stable = pct(sg, self.q_graph_stable)
        heavy = pct(sg, self.q_graph_heavy)
        if heavy <= stable:
            heavy = stable + 1e-6

        self.graph_gate.stable_thresh = stable
        self.graph_gate.threshold = heavy

        pcsd_thr = pct(sp, self.q_pcsd_heavy)
        self.pcsd.threshold = pcsd_thr

        self.calib_done = True
        print(f"[CMDA calib] graph stable={stable:.6f}, heavy={heavy:.6f}, pcsd_thr={pcsd_thr:.6f}")


    def _pcsd_k(self) -> int:
        k = int(getattr(self.args, "pcsd_k", 24))
        return max(1, min(int(self.pred_len), k))



    def _horizon_weights(self, H: int, device: torch.device) -> torch.Tensor:
        H = int(H)
        if H <= 1:
            return torch.ones(1, device=device, dtype=torch.float32)

        r_end = float(getattr(self.args, "cmda_horizon_r_end", 0.3))
        r_end = max(1e-6, min(1.0, r_end))

        lam = -float(torch.log(torch.tensor(r_end)).item()) / float(H - 1)

        h = torch.arange(H, device=device, dtype=torch.float32)
        w = torch.exp(-lam * h)

        w = w / (w.mean() + 1e-12)

        return w

    def _pcsd_loss_full_horizon(self, outputs: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        H = outputs.size(1)
        w = self._horizon_weights(H, device=outputs.device)
        w3 = w.view(1, H, 1)
        return ((outputs - y_true) ** 2 * w3).mean()

    def _weighted_mse(self, pred: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:

        w3 = w.view(1, -1, 1)
        return ((pred - y) ** 2 * w3).mean()

    def _build_model(self, model=None, framework_class=None):
        return super()._build_model(model, framework_class=MechAdapt)

    def _reset_regime_log(self):
        out_dir = str(getattr(self.args, "regime_log_root", os.path.join("results", "regime")))
        os.makedirs(out_dir, exist_ok=True)

        path = os.path.join(out_dir, f"{self.args.dataset}_pred{self.pred_len}_regime.csv")

        #delete if exists
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            print(f"[WARN] Could not delete {path}: {e}")

        return path

    def _log_regime_csv(self, step, regime, sg, sp, w_s, w_m):
        import csv
        path = getattr(self, "regime_log_path", None)
        if path is None:
            out_dir = str(getattr(self.args, "regime_log_root", os.path.join("results", "regime")))
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"{self.args.dataset}_pred{self.pred_len}_regime.csv")

        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["step", "regime", "s_graph", "s_pcsd", "w_s", "w_m"])
            writer.writerow([step, regime, sg, sp, w_s, w_m])



    def _replace_seq_x(self, batch: BatchType, new_seq_x: torch.Tensor) -> BatchType:
        if isinstance(batch, tuple):
            b = list(batch)
            b[0] = new_seq_x
            return tuple(b)
        if isinstance(batch, list):
            b = list(batch)
            b[0] = new_seq_x
            return b
        return batch

    def _apply_cra_to_batch(self, batch, update_ref: bool = True):
        seq_x = batch[0]
        if not isinstance(seq_x, torch.Tensor) or seq_x.dim() != 3:
            return batch
        if not getattr(self.args, "pin_gpu", False):
            seq_x = seq_x.to(self.device)

        with torch.no_grad():
            if update_ref:
                self.cra.update_reference(seq_x)
            if self.cra.steps < self.cra.warmup_steps:
                return batch
            seq_x2 = self.cra(seq_x)
        return self._replace_seq_x(batch, seq_x2)

    # in __init__

    def _smooth_scores(self, sg: float, sp: float):
        if self.sg_ema is None:
            self.sg_ema, self.sp_ema = sg, sp
        else:
            b = float(self.mix_ema)
            self.sg_ema = b * self.sg_ema + (1.0 - b) * sg
            self.sp_ema = b * self.sp_ema + (1.0 - b) * sp
        return self.sg_ema, self.sp_ema

    def _mix_weights(self, s_graph: float, s_pcsd: float):
        tg = float(getattr(self.graph_gate, "threshold", 0.15))
        tp = float(getattr(self.pcsd, "threshold", 0.15))

        # logits
        lg = self.mix_alpha_graph * (float(s_graph) - tg)
        lp = self.mix_alpha_pcsd * (float(s_pcsd) - tp)

        #EMA on logits
        if self.lg_ema is None:
            self.lg_ema, self.lp_ema = lg, lp
        else:
            b = float(self.mix_ema)
            self.lg_ema = b * self.lg_ema + (1.0 - b) * lg
            self.lp_ema = b * self.lp_ema + (1.0 - b) * lp

        lg_t = torch.as_tensor(self.lg_ema, device=self.device)
        lp_t = torch.as_tensor(self.lp_ema, device=self.device)

        w_graph = torch.sigmoid(lg_t)
        w_pcsd = torch.sigmoid(lp_t)

        w_m = (w_graph * w_pcsd).clamp(0.0, 1.0)
        w_s = (1.0 - w_m).clamp(0.0, 1.0)
        return w_s, w_m

    def _apply_light_adapter(self, outputs: torch.Tensor, batch) -> torch.Tensor:
        seq_x = batch[0].to(self.device) if not getattr(self.args, "pin_gpu", False) else batch[0]

        if self.causal_channel_indices is not None:
            idx = torch.as_tensor(self.causal_channel_indices, device=seq_x.device, dtype=torch.long)
            seq_x = seq_x.index_select(dim=2, index=idx)

        delta_raw = self.light_adapter(seq_x)  # [B,H,1]

        s = 1.0 if self.light_scale_ema is None else float(self.light_scale_ema)
        scale = torch.as_tensor(s, device=delta_raw.device, dtype=delta_raw.dtype)

        delta = delta_raw * scale

        # clamp is differentiable (piecewise), safe in train+infer
        dmax = float(getattr(self.args, "cmda_light_clip", 0.5))
        delta = delta.clamp(-dmax, dmax)

        if not torch.is_grad_enabled():
            #EMA on delta
            beta = float(getattr(self, "light_delta_beta", 0.90))
            if getattr(self, "light_delta_ema", None) is None or self.light_delta_ema.shape != delta.shape:
                self.light_delta_ema = delta.detach()
            else:
                self.light_delta_ema = beta * self.light_delta_ema + (1.0 - beta) * delta.detach()
            delta = self.light_delta_ema


            max_step = float(getattr(self.args, "cmda_light_delta_max_step", 0.05))
            if getattr(self, "light_delta_prev", None) is None or self.light_delta_prev.shape != delta.shape:
                self.light_delta_prev = delta.detach()
            else:
                lo = self.light_delta_prev - max_step
                hi = self.light_delta_prev + max_step
                delta = torch.max(lo, torch.min(hi, delta))
                self.light_delta_prev = delta.detach()

        #apply
        if outputs.size(-1) == 1:
            return outputs + delta

        tv = self.target_variate
        if tv is None:
            tv = outputs.size(-1) - 1
        out = outputs.clone()
        out[:, :, tv:tv + 1] = out[:, :, tv:tv + 1] + delta
        return out

    def _get_aligned_y_true(self, batch, outputs_like: torch.Tensor) -> torch.Tensor:
        y_true = batch[self.label_position]
        if not getattr(self.args, "pin_gpu", False):
            y_true = y_true.to(self.device)

        if y_true.dim() == 3:
            y_true = y_true[:, -self.pred_len:, :]


        tv = self.target_variate
        if tv is not None and y_true.size(-1) > 1:
            tv = int(max(0, min(tv, y_true.size(-1) - 1)))
            y_true = y_true[:, :, tv:tv + 1]

        return y_true

    def _maybe_init_light(self, batch):
        """
        Lazily initialize LightAdapter (net) and its optimizer once we see real seq_x.
        """
        if not self.enable_light:
            return

        seq_x = batch[0]
        if not getattr(self.args, "pin_gpu", False):
            seq_x = seq_x.to(self.device)

        if self.causal_channel_indices is not None:
            idx = torch.as_tensor(self.causal_channel_indices, device=seq_x.device, dtype=torch.long)
            seq_x = seq_x.index_select(dim=2, index=idx)


        self.light_adapter.maybe_init(seq_x)

        #init optimiser
        if self.light_optim is None:
            params = list(self.light_adapter.parameters())
            if len(params) == 0:
                raise RuntimeError("LightAdapter still has no parameters after maybe_init().")
            self.light_optim = optim.Adam(params, lr=self.light_lr)

    def _pool_factor(self):

        return int(getattr(self.args, "cmda_pool_factor", 1))

    def _x_idx(self, C: int, device):
        y_idx = int(getattr(self.args, "y_index_in_causal", -1))
        if y_idx < 0:
            y_idx = C - 1
        y_idx = max(0, min(y_idx, C - 1))
        idx = [j for j in range(C) if j != y_idx]
        return torch.as_tensor(idx, device=device, dtype=torch.long), y_idx

    def _summarize_H(self, H: torch.Tensor) -> torch.Tensor:
        # H: [L, C]
        pool = self._pool_factor()
        if pool <= 1:
            return H

        mode = str(getattr(self.args, "cmda_pool_mode", "avg")).lower()

        L, C = H.shape
        L2 = (L // pool) * pool
        Hc = H[-L2:, :]  # trim to multiple of pool
        Hc = Hc.view(L2 // pool, pool, C)  # [L', pool, C]

        if mode == "stats":
            m = Hc.mean(dim=1)
            s = Hc[:, -1, :] - Hc[:, 0, :]
            return torch.cat([m, s], dim=-1)  # [L', 2C]
        else:
            return Hc.mean(dim=1)  # [L', C]

    def _extract_causal_matrix(self, batch: BatchType) -> Optional[torch.Tensor]:
        seq_x = batch[0]
        if not isinstance(seq_x, torch.Tensor) or seq_x.dim() != 3:
            return None

        x = seq_x.detach()
        if not getattr(self.args, "pin_gpu", False):
            x = x.to(self.device)

        x = x.mean(dim=0)  # [L, C_all]

        if self.causal_channel_indices is not None:
            idx = torch.as_tensor(self.causal_channel_indices, device=x.device, dtype=torch.long)
            x = x.index_select(dim=1, index=idx)  # [L, C_causal]

        if self.causal_dim == 0:
            self.causal_dim = int(x.size(-1))

        x = self._summarize_H(x)
        return x

    def calibrate_thresholds_from_train_loader(self, train_loader):
        # reset state if possible
        if hasattr(self.graph_estimator, "reset"):
            self.graph_estimator.reset()
        if hasattr(self.graph_gate, "reset"):
            self.graph_gate.reset()
        if hasattr(self.pcsd, "reset"):
            self.pcsd.reset()

        self._model.eval()
        self.calib_done = False
        self._calib_s_graph = []
        self._calib_s_pcsd = []

        with torch.no_grad():
            pass

        max_calib_batches = int(getattr(self.args, "cmda_calib_batches", 500))

        for i, batch in enumerate(train_loader):
            if i >= max_calib_batches:
                break

            #batch = self._apply_cra_to_batch(batch)

            H = self._extract_causal_matrix(batch)
            if H is not None:
                self.graph_estimator.update_buffer(H)

            # if (len(self._calib_s_graph) % self.graph_train_every) == 0:
            if (i % self.graph_train_every) == 0:
                self.graph_estimator.train_on_buffer(
                    steps=self.graph_train_steps,
                    lr=self.graph_lr,
                    batch_size=int(getattr(self.args, "cmda_graph_batch_size", 64)),
                )

            A_t = self.graph_estimator.current_graph()

            _, s_graph = self.graph_gate.decide(A_t)
            sg = float(s_graph.item())

            sig = self._pcsd_signature(batch)
            _, s_pcsd = self.pcsd.decide(sig)
            sp = float(s_pcsd)

            self._calib_s_graph.append(sg)
            self._calib_s_pcsd.append(sp)

        self._finalize_calibration()


    def _pcsd_signature(self, batch):
        was_training = self._model.training
        old_apply = getattr(self, "_cmda_apply", False)
        self._cmda_apply = False  # <- critical

        self._model.eval()
        self._model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            outputs = super().forward(batch)  # <- use backbone forward
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            outputs = outputs[:, -self.pred_len:, :]

            y_true = batch[self.label_position].to(self.device)
            y_true = y_true[:, -self.pred_len:, :]


            if y_true.size(-1) > 1 and outputs.size(-1) == 1:
                tv = self.target_variate
                if tv is not None:
                    y_true = y_true[:, :, tv:tv + 1]

            pcsd_mode = str(getattr(self.args, "pcsd_horizon_mode", "weighted_full")).lower()
            if pcsd_mode == "k":
                k = self._pcsd_k()
                loss = self._select_criterion()(outputs[:, :k, :], y_true[:, :k, :])
            elif pcsd_mode == "full":
                loss = self._select_criterion()(outputs, y_true)
            else:
                loss = self._pcsd_loss_full_horizon(outputs, y_true)

            sig = self.pcsd.signature_from_loss(loss)

        self._model.zero_grad(set_to_none=True)
        self._model.train(was_training)
        self._cmda_apply = old_apply
        return sig


    def _apply_mechanism_adapter_delta(self, batch: BatchType) -> torch.Tensor:

        seq_x = batch[0]
        if not getattr(self.args, "pin_gpu", False):
            seq_x = seq_x.to(self.device)

        if self.causal_channel_indices is not None:
            idx = torch.as_tensor(self.causal_channel_indices, device=seq_x.device, dtype=torch.long)
            seq_x = seq_x.index_select(dim=2, index=idx)

        self.mechanism_adapter.maybe_init(seq_x)
        delta = self.mechanism_adapter(seq_x)  # [B, pred_len, 1]
        return delta

    def _add_delta_to_outputs(self, outputs: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:

        if outputs.size(-1) == 1:
            return outputs + delta

        tv = self.target_variate
        if tv is None:
            tv = outputs.size(-1) - 1
        out = outputs.clone()
        out[:, :, tv:tv + 1] = out[:, :, tv:tv + 1] + delta
        return out


    #CSV save for causal graph
    def _maybe_save_graph_csv(self, A_t: torch.Tensor):
        if self.graph_log_every <= 0:
            return
        if (self.cmda_step % self.graph_log_every) != 0:
            return

        try:
            import pandas as pd
            out_dir = os.path.join(self.graph_log_root, str(self.args.dataset))
            os.makedirs(out_dir, exist_ok=True)

            A = A_t.detach().float().cpu().numpy()
            cols = [f"lag{l+1}" for l in range(A.shape[1])]

            idx = self.causal_var_names
            if len(idx) != A.shape[0]:
                idx = [f"var{i}" for i in range(A.shape[0])]

            df = pd.DataFrame(A, index=idx, columns=cols)
            df.to_csv(os.path.join(out_dir, f"graph_step_{self.cmda_step}.csv"))
        except Exception:
            pass

    def _detach_model_state(self):

        m = self._model


        for name in [
            "recent_update",
            "recent_batch",
            "drift_buf",
            "last_graph_score",
            "last_pcsd_score",
            "last_do_update",
        ]:
            if hasattr(m, name):
                v = getattr(m, name)
                if torch.is_tensor(v):
                    setattr(m, name, v.detach())


        for name in ["state", "cache", "buffers"]:
            if hasattr(m, name):
                v = getattr(m, name)
                if isinstance(v, dict):
                    for k, t in v.items():
                        if torch.is_tensor(t):
                            v[k] = t.detach()

    def _apply_latched_control(self):

        m = self._model

        # graphs
        if hasattr(m, "set_graphs"):
            A_t = self._cmda_last_A_t
            A_ref = self._cmda_last_A_ref
            if (A_t is not None) or (A_ref is not None):
                m.set_graphs(A_t, A_ref)
            else:

                pass

        # drift
        if hasattr(m, "set_drift"):
            m.set_drift(self._cmda_last_drift)

    def _train_cra(self, batch):
        if self.cra_optim is None:
            return

        seq_x = batch[0]
        if not getattr(self.args, "pin_gpu", False):
            seq_x = seq_x.to(self.device)


        if self.cra.steps < self.cra.warmup_steps:
            return

        mu_ref = self.cra.mu_ref.detach()
        std_ref = self.cra.std_ref.detach()

        for _ in range(self.cra_steps):
            self.cra_optim.zero_grad(set_to_none=True)

            seq_x_hat = self.cra(seq_x)  # transformed

            mu_hat = seq_x_hat.mean(dim=(0, 1))
            std_hat = seq_x_hat.std(dim=(0, 1), unbiased=False).clamp_min(self.cra.eps)

            loss = (mu_hat - mu_ref).pow(2).mean() + (std_hat - std_ref).pow(2).mean()
            loss.backward()
            self.cra_optim.step()


    def update_valid(self):
        old = getattr(self, "phase", "test")
        try:
            self.phase = "valid"
            return super().update_valid()
        finally:
            self.phase = old

    def _forward_base(self, batch):

        self._model.eval()
        with torch.no_grad():
            out = super().forward(batch)
            if isinstance(out, (tuple, list)):
                out = out[0]
            if out.dim() == 3:
                out = out[:, -self.pred_len:, :]
            return out

    def forward(self, batch):
        # if self._model.training and (
        #         getattr(self, "_cmda_apply", False) or getattr(self, "_cmda_last_regime", None) is not None):
        #     print("[WARN] CMDA flags set during training!", self._cmda_apply, getattr(self, "_cmda_last_regime", None))


        phase = str(getattr(self, "phase", "train")).lower()
        if phase in ["train", "val", "vali", "valid", "offline"]:
            return super().forward(batch)

        if self._model.training:
            return super().forward(batch)

        # If not in the online stream, also pure backbone
        if not bool(getattr(self, "_cmda_apply", False)):
            return super().forward(batch)

        if self.enable_cra:
            batch2 = self._cra_apply_to_batch_current(batch, allow_update_ref=False)
        else:
            batch2 = batch



        out = super().forward(batch2)
        out0 = out[0] if isinstance(out, (tuple, list)) else out

        if isinstance(out0, torch.Tensor) and out0.dim() == 3:
            out0 = out0[:, -self.pred_len:, :]

        regime = getattr(self, "_cmda_last_regime", "none")
        w_m = float(getattr(self, "_cmda_last_w_m", 0.0))

        if regime == "light":
            out_light = self._apply_light_adapter(out0, batch2)  # out0 + delta
            w_s = float(getattr(self, "_cmda_last_w_s", 1.0))  # store it like w_m
            w_s_t = torch.as_tensor(w_s, device=out0.device, dtype=out0.dtype)
            out_corr = out0 + w_s_t * (out_light - out0)  # out0 + w_s*delta

        elif regime == "heavy":
            # delta_m = self._apply_mechanism_adapter_delta(batch2)
            # w_m_t = torch.as_tensor(w_m, device=out0.device, dtype=out0.dtype)
            # out_corr = self._add_delta_to_outputs(out0, w_m_t * delta_m)
            out_corr = out0
        else:
            out_corr = out0

        if isinstance(out, (tuple, list)):
            out_list = list(out)
            out_list[0] = out_corr
            return type(out)(out_list)

        return out_corr

    def online(self, online_data=None, target_variate=100000, phase='test', show_progress=False):
        import numpy as np
        from tqdm import tqdm
        import torch
        from data_provider.data_factory import get_dataset, get_dataloader
        from data_provider.data_loader import Dataset_Recent
        from util.metrics import update_metrics, calculate_metrics


        self.pred_list = []
        self.true_list = []
        self.phase = phase

        if str(phase).lower() in ["val", "vali", "valid", "train", "offline"]:
            return super().online(
                online_data=online_data,
                target_variate=target_variate,
                phase=phase,
                show_progress=show_progress
            )

        if online_data is None:
            online_data = get_dataset(
                self.args, phase, self.device,
                wrap_class=self.args.wrap_data_class + [Dataset_Recent],
                **getattr(self, "wrap_data_kwargs", {})
            )

        online_loader = get_dataloader(online_data, self.args, flag='online')
        if phase == 'test' or show_progress:
            online_loader = tqdm(online_loader, mininterval=10)

        statistics = {k: 0 for k in ['total', 'y_sum', 'MSE', 'MAE', 'SMAPE']}
        model_optim = self.model_optim
        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        predictions, trues = ([], []) if self.args.do_predict else (None, None)


        recent_train_every = int(getattr(self.args, "cmda_recent_train_every", 1))
        warmup = int(getattr(self.args, "cmda_warmup_steps", 0))

        for i, (recent_data, current_data) in enumerate(online_loader):
            self.cmda_step += 1


            allow_train = (self.cmda_step > warmup) and ((self.cmda_step % recent_train_every) == 0)


            self._model.eval()  #stable graph + pcsd signature
            _loss_r, _ = self._update_online(
                recent_data, criterion, model_optim, scaler,
                flag_current=False,
                allow_train=allow_train
            )

            self._model.eval()
            old_apply = getattr(self, "_cmda_apply", False)
            try:
                with torch.no_grad():

                    self._apply_latched_control()

                    self._cmda_apply = True
                    outputs_eval = self.forward(current_data)


                    if isinstance(outputs_eval, (tuple, list)):
                        outputs_eval = outputs_eval[0]
                    if outputs_eval.dim() == 3:
                        outputs_eval = outputs_eval[:, -self.pred_len:, :]

                    #pick target channel, we do MS
                    if outputs_eval.size(-1) > 1:
                        tv = self.target_variate
                        if tv is None:
                            tv = outputs_eval.size(-1) - 1
                        tv = int(max(0, min(tv, outputs_eval.size(-1) - 1)))
                        outputs_eval = outputs_eval[:, :, tv:tv + 1]

                    y_true_eval = self._get_aligned_y_true(current_data, outputs_eval)


                    # LOG (current)
                    # if self.pt_log_every > 0 and (self.cmda_step % self.pt_log_every) == 0:
                    #     self._log_pred_true_row(
                    #         step=self.cmda_step,
                    #         outputs=outputs_eval,
                    #         y_true=y_true_eval,
                    #         regime=getattr(self, "_cmda_last_regime", "none"),
                    #         sg=float(self.sg_ema if self.sg_ema is not None else 0.0),
                    #         sp=float(self.sp_ema if self.sp_ema is not None else 0.0),
                    #         w_s=float(getattr(self, "_cmda_last_w_s", 1.0)),
                    #         w_m=float(getattr(self, "_cmda_last_w_m", 0.0)),
                    #     )

                    update_metrics(outputs_eval, y_true_eval, statistics, getattr(self.args, 'target_variate'))

                    if self.args.do_predict:
                        predictions.append(outputs_eval.detach().cpu().numpy())
                        trues.append(y_true_eval.detach().cpu().numpy())
            finally:
                self._cmda_apply = old_apply

        metrics = calculate_metrics(statistics)
        rmse, mae, smape = metrics['RMSE'], metrics['MAE'], metrics['SMAPE']
        if phase == 'test':
            print(f'rmse:{rmse}, mae:{mae}, smape:{smape}')

        if self.args.do_predict:
            self.pred_list = predictions
            self.true_list = trues
            return rmse, mae, smape, online_data, predictions
        else:
            return rmse, mae, smape, online_data

    def _adapt_on_recent(self, batch_recent, criterion, optimizer, scaler=None, allow_train: bool = True):

        self._cmda_last_regime = "none"
        self._cmda_last_w_s = 1.0
        self._cmda_last_w_m = 0.0

        old_apply = getattr(self, "_cmda_apply", False)
        try:

            self._cmda_apply = False


            if self.enable_cra:
                self._cra_update_ref_from_recent(batch_recent)
                batch_cra = self._cra_apply_to_batch_current(batch_recent, allow_update_ref=False)
            else:
                batch_cra = batch_recent



            H = self._extract_causal_matrix(batch_cra)
            if H is not None:
                self.graph_estimator.update_buffer(H)


            if (self.cmda_step % self.graph_train_every) == 0:
                self.graph_estimator.train_on_buffer(
                    steps=self.graph_train_steps,
                    lr=self.graph_lr,
                    batch_size=int(getattr(self.args, "cmda_graph_batch_size", 128)),
                )


            A_t_raw = self.graph_estimator.current_graph()
            A_t_norm = self.graph_gate._normalize_A(A_t_raw)

            betaA = float(getattr(self, "A_ema_beta", 0.95))
            if (self.A_ema is None) or (self.A_ema.shape != A_t_norm.shape):
                self.A_ema = A_t_norm.detach()
            else:
                self.A_ema = betaA * self.A_ema + (1.0 - betaA) * A_t_norm.detach()
            A_t = self.A_ema

            self._maybe_save_graph_csv(A_t)

            do_graph_heavy, s_graph = self.graph_gate.decide(A_t)
            A_ref = getattr(self.graph_gate, "ref", None)
            sg = float(s_graph.item())

            if hasattr(self._model, "set_drift") and (A_ref is not None):
                drift = self._model.encode_graph_delta(A_t, A_ref).detach()
                self._model.set_drift(drift)
            else:
                self._model.set_drift(None)

            if hasattr(self._model, "set_graphs"):
                self._model.set_graphs(A_t, A_ref)


            self._cmda_last_A_t = A_t.detach() if torch.is_tensor(A_t) else None
            If_A_ref = A_ref.detach() if torch.is_tensor(A_ref) else None
            self._cmda_last_A_ref = If_A_ref
            self._cmda_last_have_graphs = (self._cmda_last_A_t is not None)


            if (A_ref is not None) and hasattr(self._model, "encode_graph_delta"):
                self._cmda_last_drift = self._model.encode_graph_delta(A_t, A_ref).detach()
            else:
                self._cmda_last_drift = None


            if sg < float(self.graph_gate.stable_thresh):
                graph_regime = "light"
            elif sg > float(self.graph_gate.threshold):
                graph_regime = "heavy"
            else:
                graph_regime = "none"


            pcsd_every = int(getattr(self.args, "cmda_pcsd_every", 10))
            pcsd_every = max(1, pcsd_every)

            if (graph_regime == "heavy") or (self.cmda_step % pcsd_every == 0):
                sig = self._pcsd_signature(batch_cra)  # costly
                do_pcsd_heavy, s_pcsd = self.pcsd.decide(sig)
                sp = float(s_pcsd)
                self.last_sp = sp
            else:
                do_pcsd_heavy = False
                sp = float(self.last_sp) if self.last_sp is not None else float(getattr(self.pcsd, "threshold", 0.15))


            sg_s, sp_s = self._smooth_scores(sg, sp)
            w_s, w_m = self._mix_weights(sg_s, sp_s)
            w_s = float(w_s.item()) if isinstance(w_s, torch.Tensor) else float(w_s)
            w_m = float(w_m.item()) if isinstance(w_m, torch.Tensor) else float(w_m)

            self._sg_buf.append(float(sg))
            self._sp_buf.append(float(sp))

            if self.dyn_thresh and (self.cmda_step % self.recalc_every == 0):
                self._update_dynamic_thresholds()


            w_s = float(w_s.item()) if isinstance(w_s, torch.Tensor) else float(w_s)
            w_m = float(w_m.item()) if isinstance(w_m, torch.Tensor) else float(w_m)


            if graph_regime == "heavy" and bool(do_pcsd_heavy):
                regime = "heavy"
            elif graph_regime == "light" and self.enable_light:
                regime = "light"
            else:
                regime = "none"

            self._cmda_last_regime = regime
            self._cmda_last_w_s = w_s
            self._cmda_last_w_m = w_m

            prev = getattr(self, "_last_regime", "none")
            if prev != "light" and regime == "light":
                self.light_delta_ema = None
                self.light_delta_prev = None
            if prev == "light" and regime != "light":
                self.light_delta_ema = None
                self.light_delta_prev = None
            self._last_regime = regime


            if hasattr(self._model, "last_graph_score"):
                self._model.last_graph_score = torch.as_tensor(sg, device=self.device).detach()
            if hasattr(self._model, "last_pcsd_score"):
                self._model.last_pcsd_score = torch.as_tensor(sp, device=self.device).detach()
            if hasattr(self._model, "last_regime"):
                self._model.last_regime = regime


            log_every = int(getattr(self.args, "regime_log_every", 1))
            if log_every > 0 and (self.cmda_step % log_every) == 0:
                self._log_regime_csv(self.cmda_step, regime, sg, sp, w_s, w_m)


            #clear
            if hasattr(self._model, "set_drift"):
                self._model.set_drift(None)

            if not allow_train:

                return torch.tensor(0.0, device=self.device), None


            if regime == "heavy":

                self._cmda_apply = False

                self._set_cmda_train_targets(heavy=True)
                self._model.train()

                if hasattr(self._model, "set_drift"):
                    if (A_ref is not None) and hasattr(self._model, "encode_graph_delta"):
                        drift = self._model.encode_graph_delta(A_t, A_ref).detach()
                        self._model.set_drift(drift)
                    else:
                        self._model.set_drift(None)

                loss_h, _ = super()._update_online(batch_cra, criterion, optimizer, scaler, flag_current=True)
                if self.cmda_step % 200 == 0:
                    gen_params = list(self._model.generator.parameters())
                    opt_params = [p for g in optimizer.param_groups for p in g["params"]]
                    opt_ids = {id(p) for p in opt_params}
                    inter = sum(id(p) in opt_ids for p in gen_params)
                    # print(f"[OPT dbg PRE] generator_in_opt={inter}/{len(gen_params)} "
                    #       f"req_grad={sum(p.requires_grad for p in gen_params)}", flush=True)

                return loss_h, None



            if regime == "light":
                base = self._forward_base(batch_cra).detach()

                self._maybe_init_light(batch_cra)

                with torch.no_grad():
                    seq_x = batch_cra[0]
                    if not getattr(self.args, "pin_gpu", False):
                        seq_x = seq_x.to(self.device)

                    if self.causal_channel_indices is not None:
                        idx = torch.as_tensor(self.causal_channel_indices, device=seq_x.device, dtype=torch.long)
                        seq_x = seq_x.index_select(dim=2, index=idx)

                    delta_raw = self.light_adapter(seq_x).detach()
                    self._update_light_scale_from_recent(base, batch_cra, delta_raw)


                self.light_adapter.train()
                self._model.eval()

                y_true = self._get_aligned_y_true(batch_cra, base)  # uses recent labels (ok)
                w = self._horizon_weights(self.pred_len, device=y_true.device)
                w_s_t = torch.as_tensor(w_s, device=base.device, dtype=base.dtype)

                last_loss = None
                for _ in range(self.light_steps):
                    self.light_optim.zero_grad(set_to_none=True)
                    out_light = self._apply_light_adapter(base, batch_cra)
                    delta = (out_light - base)
                    lam_time = float(getattr(self.args, "cmda_light_time_lam", 2.0))  # start 1–5
                    loss_time = torch.tensor(0.0, device=delta.device)

                    if self._light_prev_delta_train is not None and self._light_prev_delta_train.shape == delta.shape:
                        loss_time = (delta - self._light_prev_delta_train).pow(2).mean()

                    self._light_prev_delta_train = delta.detach()

                    pred_train = base + w_s_t * delta
                    loss_fit= self._weighted_mse(pred_train, y_true, w)


                    #smoothness
                    lam_smooth = float(getattr(self.args, "cmda_light_smooth_lam", 0.1))
                    loss_smooth = (delta[:, 1:, :] - delta[:, :-1, :]).pow(2).mean()

                    #mag penalty
                    lam_mag = float(getattr(self.args, "cmda_light_mag_lam", 0.01))
                    loss_mag = delta.pow(2).mean()

                    loss = loss_fit + lam_time * loss_time + lam_smooth * loss_smooth + lam_mag * loss_mag


                    loss.backward()
                    self.light_optim.step()
                    last_loss = loss.detach()
                    # if self.cmda_step % 50 == 0:
                    #     with torch.no_grad():
                    #         d = delta.detach()
                    #         print(f"[LIGHT dbg] step={self.cmda_step} w_s={w_s:.3f} "
                    #               f"|delta|_mean={d.abs().mean().item():.4f} "
                    #               f"|delta|_max={d.abs().max().item():.4f} "
                    #               f"loss_fit={loss_fit.item():.4f} "
                    #               f"ls={loss_smooth.item():.4f} lm={loss_mag.item():.4f}",
                    #               flush=True)

                self.light_adapter.eval()
                return (last_loss if last_loss is not None else torch.tensor(0.0, device=self.device)), None


            return torch.tensor(0.0, device=self.device), None

        finally:
            self._cmda_apply = old_apply

    def _update_online(self, batch, criterion, optimizer, scaler=None, flag_current=False, allow_train: bool = True):

        phase = str(getattr(self, "phase", "train")).lower()
        if phase in ["train", "offline", "val", "vali", "valid"]:
            self._cmda_apply = False
            return super()._update_online(batch, criterion, optimizer, scaler, flag_current=True)


        if flag_current:
            old_apply = getattr(self, "_cmda_apply", False)
            try:
                self._model.eval()
                self._cmda_apply = True

                with torch.no_grad():
                    out = self.forward(batch)
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    if isinstance(out, torch.Tensor) and out.dim() == 3:
                        out = out[:, -self.pred_len:, :]


                return torch.tensor(0.0, device=self.device), out
            finally:
                self._cmda_apply = old_apply


        return self._adapt_on_recent(batch, criterion, optimizer, scaler, allow_train=allow_train)