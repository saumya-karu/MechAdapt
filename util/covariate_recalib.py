#util/covariate_recalib.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class CovariateRecalibration(nn.Module):

    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-5,
        learn_affine: bool = True,
        clip_gamma: float = 3.0,
        ref_ema: float = 0.999,
        warmup_steps: int = 50,
        device=None,
    ):
        super().__init__()
        self.C = int(num_channels)
        self.eps = float(eps)
        self.learn_affine = bool(learn_affine)
        self.clip_gamma = float(clip_gamma)
        self.ref_ema = float(ref_ema)
        self.warmup_steps = int(warmup_steps)

        #reference moments
        self.register_buffer("mu_ref", torch.zeros(self.C), persistent=True)
        self.register_buffer("std_ref", torch.ones(self.C), persistent=True)
        self.register_buffer("steps", torch.zeros((), dtype=torch.long), persistent=True)
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool), persistent=True)


        if self.learn_affine:
            self.gamma = nn.Parameter(torch.ones(self.C))
            self.beta = nn.Parameter(torch.zeros(self.C))
        else:
            self.gamma = None
            self.beta = None

        if device is not None:
            self.to(device)


    def _compute_moments_grad(self, x: torch.Tensor):

        mu = x.mean(dim=(0, 1))
        var = x.var(dim=(0, 1), unbiased=False)
        std = torch.sqrt(var + self.eps)
        return mu, std

    @torch.no_grad()
    def _compute_moments_nograd(self, x: torch.Tensor):

        mu = x.mean(dim=(0, 1))
        var = x.var(dim=(0, 1), unbiased=False)
        std = torch.sqrt(var + self.eps)
        return mu, std


    @torch.no_grad()
    def update_reference(self, x: torch.Tensor):

        if not isinstance(x, torch.Tensor) or x.dim() != 3:
            return

        mu, std = self._compute_moments_nograd(x)
        self.steps += 1

        if not bool(self.initialized.item()):
            self.mu_ref.copy_(mu)
            self.std_ref.copy_(std)
            self.initialized.fill_(True)
            return

        ema = self.ref_ema
        if int(self.steps.item()) <= self.warmup_steps:
            ema = 0.95

        self.mu_ref.mul_(ema).add_(mu * (1.0 - ema))
        self.std_ref.mul_(ema).add_(std * (1.0 - ema))

    def forward(self, x: torch.Tensor):

        if not isinstance(x, torch.Tensor) or x.dim() != 3:
            return x


        if not bool(self.initialized.item()):
            return x

        mu_cur = x.mean(dim=(0, 1))
        std_cur = torch.sqrt(x.var(dim=(0, 1), unbiased=False) + self.eps)


        x_norm = (x - mu_cur.view(1, 1, -1)) / std_cur.view(1, 1, -1)
        x_rec = x_norm * self.std_ref.view(1, 1, -1) + self.mu_ref.view(1, 1, -1)

        if self.learn_affine:
            gamma = torch.clamp(self.gamma, 1.0 / self.clip_gamma, self.clip_gamma)
            beta = self.beta
            x_rec = x_rec * gamma.view(1, 1, -1) + beta.view(1, 1, -1)

        return x_rec


    def moment_loss(self, x: torch.Tensor):

        if not isinstance(x, torch.Tensor) or x.dim() != 3:

            return torch.tensor(0.0, device=x.device if isinstance(x, torch.Tensor) else "cpu")

        if not bool(self.initialized.item()):

            return x.sum() * 0.0

        mu, std = self._compute_moments_grad(x)
        return F.mse_loss(mu, self.mu_ref) + F.mse_loss(std, self.std_ref)

