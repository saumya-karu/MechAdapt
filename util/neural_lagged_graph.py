
# util/neural_lagged_graph.py


from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _LagPredictor(nn.Module):

    def __init__(self, C_in: int, lag_len: int, hidden: int = 64, use_pos_emb: bool = True):
        super().__init__()
        self.C_in = C_in
        self.lag_len = lag_len
        self.hidden = hidden
        self.use_pos_emb = use_pos_emb


        if use_pos_emb:
            self.pos_emb = nn.Parameter(torch.zeros(1, 1, lag_len))
            nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        else:
            self.pos_emb = None


        self.mix1 = nn.Conv1d(in_channels=C_in, out_channels=hidden, kernel_size=1)
        self.mix2 = nn.Conv1d(in_channels=hidden, out_channels=hidden, kernel_size=1)


        self.head = nn.Conv1d(in_channels=hidden, out_channels=1, kernel_size=1)


        self.lag_gate = nn.Parameter(torch.zeros(1, 1, lag_len))
        nn.init.zeros_(self.lag_gate)

    def forward(self, x_lags: torch.Tensor) -> torch.Tensor:

        h = x_lags
        if self.pos_emb is not None:
            h = h + self.pos_emb

        h = F.relu(self.mix1(h))
        h = F.relu(self.mix2(h))


        y_per_lag = self.head(h)

        w = F.softplus(self.lag_gate)
        yhat = (y_per_lag * w).sum(dim=-1)
        return yhat


class NeuralLaggedCausalGraph:
    def __init__(
        self,
        tau_max: int = 12,
        window: int = 512,
        y_index: int = -1,
        device: Optional[torch.device] = None,
        hidden: int = 64,


        include_lag0: bool = True,
        x_only_inputs: bool = True,
        use_pos_emb: bool = True,
    ):
        self.tau_max = int(tau_max)
        self.window = int(window)
        self.y_index = int(y_index)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden = hidden

        self.include_lag0 = bool(include_lag0)
        self.x_only_inputs = bool(x_only_inputs)
        self.use_pos_emb = bool(use_pos_emb)

        self.buf = None
        self.model = None
        self.C_all = None
        self.C_in = None

    def update_buffer(self, H: torch.Tensor):

        H = H.detach().to(self.device)

        if self.buf is None:
            self.buf = H
        else:
            self.buf = torch.cat([self.buf, H], dim=0)

        if self.buf.size(0) > self.window:
            self.buf = self.buf[-self.window:, :]

        if self.C_all is None:
            self.C_all = int(self.buf.size(-1))
            self._maybe_init_model()

    def _maybe_init_model(self):
        if self.C_all is None:
            return
        if self.model is None:
            y_idx = self._y_idx()
            if self.x_only_inputs:
                self.C_in = self.C_all - 1  #drop Y from inputs
            else:
                self.C_in = self.C_all

            lag_len = self.tau_max + 1 if self.include_lag0 else self.tau_max
            self.model = _LagPredictor(
                C_in=self.C_in, lag_len=lag_len, hidden=self.hidden, use_pos_emb=self.use_pos_emb
            ).to(self.device)

    def _y_idx(self) -> int:
        if self.buf is None:
            return -1
        return self.y_index if self.y_index >= 0 else (self.buf.size(1) + self.y_index)

    def _build_lag_tensor(self, t_indices: torch.Tensor) -> torch.Tensor:

        T = self.buf.size(0)
        C_all = self.buf.size(1)
        y_idx = self._y_idx()

        if self.x_only_inputs:

            cols = [i for i in range(C_all) if i != y_idx]
            buf_in = self.buf[:, cols]  # [T, C_in]
        else:
            buf_in = self.buf
        C_in = buf_in.size(1)

        tau = self.tau_max
        if self.include_lag0:
            min_t = tau
            t_indices = t_indices.clamp(min=min_t, max=T - 1)

            lags = []
            for l in range(0, tau + 1):
                idx = t_indices - l
                lags.append(buf_in.index_select(dim=0, index=idx))
            X = torch.stack(lags, dim=1).transpose(1, 2).contiguous()
        else:
            min_t = tau
            t_indices = t_indices.clamp(min=min_t, max=T - 1)

            lags = []
            for l in range(1, tau + 1):
                idx = t_indices - l
                lags.append(buf_in.index_select(dim=0, index=idx))
            X = torch.stack(lags, dim=1).transpose(1, 2).contiguous()
        return X

    def train_on_buffer(self, steps: int = 20, lr: float = 1e-3, batch_size: int = 64):

        if self.buf is None or self.buf.size(0) < (self.tau_max + 5):
            return
        self._maybe_init_model()
        if self.model is None:
            return

        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=float(lr))

        T = self.buf.size(0)
        y_idx = self._y_idx()

        for _ in range(int(steps)):
            t = torch.randint(low=self.tau_max, high=T, size=(int(batch_size),), device=self.device)
            X_lags = self._build_lag_tensor(t)
            y = self.buf.index_select(dim=0, index=t)[:, y_idx:y_idx + 1]

            yhat = self.model(X_lags)  #[B, 1]
            loss = F.mse_loss(yhat, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        self.model.eval()

    def current_graph(self) -> torch.Tensor:

        if self.buf is None or self.model is None:
            C = self.C_in or 1
            L = self.tau_max + 1 if self.include_lag0 else self.tau_max
            return torch.zeros((C, L), device=self.device)

        self.model.eval()
        T = self.buf.size(0)
        if T < (self.tau_max + 5):
            C = self.C_in or 1
            L = self.tau_max + 1 if self.include_lag0 else self.tau_max
            return torch.zeros((C, L), device=self.device)

        B = min(64, max(16, T // 8))
        t = torch.randint(low=self.tau_max, high=T, size=(B,), device=self.device)

        with torch.enable_grad():
            X_lags = self._build_lag_tensor(t).detach().clone()
            X_lags.requires_grad_(True)

            yhat = self.model(X_lags)
            s = yhat.sum()

            grads = torch.autograd.grad(
                s, X_lags, retain_graph=False, create_graph=False, allow_unused=False
            )[0]

            A = (grads * X_lags).abs().mean(dim=0)
            A = torch.log1p(A)

        return A.detach()


# # util/neural_lagged_graph.py
# """
# NeuralLaggedCausalGraph (TCDF-like, nonlinear)
# ----------------------------------------------
# Goal: estimate lagged causal influence strengths from each channel to Y.
#
# We maintain a rolling buffer of observed history:
#   buffer: [T, C] where columns are [X..., Y] in causal_channel_indices order.
#
# We train a small nonlinear model to predict Y_t from lagged inputs:
#   - per-source temporal convolution over lags (depthwise-style)
#   - nonlinearity
#   - output: contribution scores for each source-lag pair
#
# Graph representation A:
#   A[i, l] ~ strength of X_i at lag l influencing Y (nonnegative)
#
# This is not a full DAG guarantee (TCDF isn't either), but it is:
#   - nonlinear
#   - time-lag aware
#   - robust enough for drift detection
#   - cheap to run online (small net, few steps)
# """
#
# from typing import Optional
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class _LagPredictor(nn.Module):
#     def __init__(self, C: int, tau_max: int, hidden: int = 64):
#         super().__init__()
#         self.C = C
#         self.tau_max = tau_max
#
#         # Input will be [B, C, tau_max] (lags for each channel)
#         self.conv1 = nn.Conv1d(in_channels=C, out_channels=hidden, kernel_size=1)
#         self.conv2 = nn.Conv1d(in_channels=hidden, out_channels=hidden, kernel_size=1)
#         self.head = nn.Conv1d(in_channels=hidden, out_channels=1, kernel_size=1)
#
#     def forward(self, x_lags: torch.Tensor) -> torch.Tensor:
#         # x_lags: [B, C, tau_max]
#         h = F.relu(self.conv1(x_lags))
#         h = F.relu(self.conv2(h))
#         yhat = self.head(h)  # [B, 1, tau_max]
#         # Aggregate across lags to predict y_t (scalar)
#         # Use learned weights implicitly through convs; sum across lags
#         yhat = yhat.sum(dim=-1)  # [B, 1]
#         return yhat
#
#
# class NeuralLaggedCausalGraph:
#     def __init__(
#         self,
#         tau_max: int = 12,
#         window: int = 512,
#         y_index: int = -1,
#         device: Optional[torch.device] = None,
#         hidden: int = 64,
#     ):
#         self.tau_max = int(tau_max)
#         self.window = int(window)
#         self.y_index = int(y_index)
#         self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.hidden = hidden
#
#         self.buf = None  # tensor [T, C]
#         self.model = None
#         self.C = None
#
#     def update_buffer(self, H: torch.Tensor):
#         """
#         H: [L, C] from latest observed seq_x chunk
#         We append it to internal buffer and keep last `window` points.
#         """
#         H = H.detach().to(self.device)
#
#         if self.buf is None:
#             self.buf = H
#         else:
#             self.buf = torch.cat([self.buf, H], dim=0)
#
#         # keep last window points
#         if self.buf.size(0) > self.window:
#             self.buf = self.buf[-self.window:, :]
#
#         if self.C is None:
#             self.C = int(self.buf.size(-1))
#             self._maybe_init_model()
#
#     def _maybe_init_model(self):
#         if self.C is None:
#             return
#         if self.model is None:
#             self.model = _LagPredictor(C=self.C, tau_max=self.tau_max, hidden=self.hidden).to(self.device)
#
#     def _build_lag_tensor(self, t_indices: torch.Tensor) -> torch.Tensor:
#         """
#         Build lag tensor X_lags for each t in t_indices:
#           X_lags[b, c, l] = buf[t - (l+1), c]
#         so l=0 corresponds to lag 1.
#         """
#         # buf: [T, C]
#         T = self.buf.size(0)
#         C = self.buf.size(1)
#         tau = self.tau_max
#
#         # Ensure valid indices: t >= tau_max
#         t_indices = t_indices.clamp(min=tau)
#
#         # gather
#         lags = []
#         for l in range(1, tau + 1):
#             idx = t_indices - l
#             lags.append(self.buf.index_select(dim=0, index=idx))  # [B, C]
#         # stack over lag dimension -> [B, tau, C] then transpose -> [B, C, tau]
#         X = torch.stack(lags, dim=1).transpose(1, 2).contiguous()
#         return X
#
#     def train_on_buffer(self, steps: int = 20, lr: float = 1e-3, batch_size: int = 64):
#         """
#         Train the lag predictor to predict Y_t from past lags.
#         """
#         if self.buf is None or self.buf.size(0) < (self.tau_max + 5):
#             return
#         self._maybe_init_model()
#         if self.model is None:
#             return
#
#         self.model.train()
#         opt = torch.optim.Adam(self.model.parameters(), lr=float(lr))
#
#         T = self.buf.size(0)
#         y_idx = self.y_index if self.y_index >= 0 else (self.buf.size(1) + self.y_index)
#
#         for _ in range(int(steps)):
#             # sample t indices (predict y at t)
#             t = torch.randint(low=self.tau_max, high=T, size=(int(batch_size),), device=self.device)
#             X_lags = self._build_lag_tensor(t)  # [B, C, tau]
#             y = self.buf.index_select(dim=0, index=t)[:, y_idx:y_idx+1]  # [B, 1]
#
#             yhat = self.model(X_lags)  # [B, 1]
#             loss = F.mse_loss(yhat, y)
#
#             opt.zero_grad(set_to_none=True)
#             loss.backward()
#             opt.step()
#
#         self.model.eval()
#
#     def current_graph(self) -> torch.Tensor:
#         """
#         Return A: [C, tau_max], nonnegative influence proxy.
#         Derived from input saliency: |d yhat / d X_lags| averaged over samples.
#         """
#         if self.buf is None or self.model is None:
#             C = self.C or 1
#             return torch.zeros((C, self.tau_max), device=self.device)
#
#         self.model.eval()
#
#         T = self.buf.size(0)
#         if T < (self.tau_max + 5):
#             return torch.zeros((self.C, self.tau_max), device=self.device)
#
#         # sample a small set of t points
#         B = min(64, max(16, T // 8))
#         t = torch.randint(low=self.tau_max, high=T, size=(B,), device=self.device)
#
#         # IMPORTANT: enable grad for saliency
#         with torch.enable_grad():
#             X_lags = self._build_lag_tensor(t).detach().clone()
#             X_lags.requires_grad_(True)
#
#             yhat = self.model(X_lags)  # [B,1]
#             s = yhat.sum()
#
#             grads = torch.autograd.grad(
#                 s, X_lags, retain_graph=False, create_graph=False, allow_unused=False
#             )[0]  # [B,C,tau]
#
#             A = (grads * X_lags).abs().mean(dim=0)  # [C,tau]
#             A = torch.log1p(A)
#
#             # # drop Y row (keep X rows only)
#             # y_idx = self.y_index if self.y_index >= 0 else (A.size(0) + self.y_index)
#             # y_idx = max(0, min(y_idx, A.size(0) - 1))
#             # A = torch.cat([A[:y_idx], A[y_idx + 1:]], dim=0)  # [C-1, tau]
#
#             # row-normalize
#             #A = A / (A.sum(dim=1, keepdim=True) + 1e-12)
#
#         return A.detach()

