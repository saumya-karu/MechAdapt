# util/mechanism_adapter.py
import torch
import torch.nn as nn

class MechanismAdapter(nn.Module):
    def __init__(self, input_dim, pred_len, hidden=64, dropout=0.0, use_window=False, window=6):
        super().__init__()
        self.input_dim = int(input_dim)
        self.pred_len = int(pred_len)
        self.hidden = int(hidden)
        self.use_window = use_window
        self.window = window
        self.dropout = float(dropout)
        self.net = None

    def build(self, device=None, dtype=None):
        net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden, self.pred_len),
        )
        if device is not None or dtype is not None:
            net = net.to(device=device, dtype=dtype)
        self.net = net

    def maybe_init(self, seq_x: torch.Tensor):

        if self.net is None:
            self.build(device=seq_x.device, dtype=seq_x.dtype)

    def forward(self, seq_x: torch.Tensor):

        if seq_x.device != next(self.parameters(), seq_x).device:

            pass


        self.maybe_init(seq_x)


        if self.net is not None and next(self.net.parameters()).device != seq_x.device:
            self.net = self.net.to(device=seq_x.device, dtype=seq_x.dtype)


        x = seq_x.mean(dim=1)

        y = self.net(x)
        return y.unsqueeze(-1)


