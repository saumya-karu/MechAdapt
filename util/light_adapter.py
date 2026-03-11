import torch
import torch.nn as nn

class LightAdapter(nn.Module):

    def __init__(self, input_dim=None, pred_len=24, hidden=32, dropout=0.01, window=64, kernel=3):
        super().__init__()
        self.pred_len = int(pred_len)
        self.window = int(window)
        self.input_dim = input_dim
        self.hidden = int(hidden)
        self.dropout = float(dropout)
        self.kernel = int(kernel)

        self.net = None

    def maybe_init(self, seq_x: torch.Tensor):
        if self.net is not None:
            return
        if not isinstance(seq_x, torch.Tensor) or seq_x.dim() != 3:
            raise ValueError(f"expects [B,L,C], got {type(seq_x)} shape={getattr(seq_x,'shape',None)}")

        C = int(seq_x.size(-1))
        self.input_dim = C

        self.net = nn.Sequential(
            nn.Conv1d(C, self.hidden, kernel_size=self.kernel, padding=self.kernel // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Conv1d(self.hidden, self.hidden, kernel_size=self.kernel, padding=self.kernel // 2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.hidden, self.pred_len),
        ).to(seq_x.device)

    def forward(self, seq_x: torch.Tensor) -> torch.Tensor:
        self.maybe_init(seq_x)
        if seq_x.size(1) < 1:
            B = int(seq_x.size(0))
            return torch.zeros((B, self.pred_len, 1), device=seq_x.device, dtype=seq_x.dtype)

        w = min(self.window, int(seq_x.size(1)))
        xw = seq_x[:, -w:, :]              #[B,w,C]
        xw = xw.transpose(1, 2).contiguous()
        delta_h = self.net(xw)
        return delta_h.unsqueeze(-1)

