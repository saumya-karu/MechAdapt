# adapter/MechAdapt_cmda.py
import collections
import typing
import torch
import torch.nn as nn
import transformers

from adapter.module.base import Adaptation
from adapter.module.generator import AdaptGenerator
from adapter.module import down_up


def clip(x, max_norm=1):
    x_norm = torch.norm(x, dim=-1, keepdim=True)
    scale = torch.clip(max_norm / (x_norm + 1e-12), max=1)
    return x * scale, x_norm

class GraphConceptEncoder(nn.Module):
    def __init__(self, concept_dim, hidden=24, dropout=0.0):
        super().__init__()
        self.concept_dim = concept_dim
        self.hidden = hidden
        self.dropout = dropout
        self._built = False
        self.in_dim = None
        self.net = None

    def maybe_build(self, in_dim, device):
        in_dim = int(in_dim)
        if self._built:
            if self.in_dim != in_dim:
                raise RuntimeError(f"in_dim changed {self.in_dim}->{in_dim}")
            self.net.to(device)
            return
        self.in_dim = in_dim
        # self.net = nn.Sequential(
        #     nn.Linear(in_dim, self.hidden), nn.GELU(),
        #     nn.Dropout(self.dropout),
        #     nn.Linear(self.hidden, self.concept_dim),
        # ).to(device)

        self.net = nn.Sequential(
            nn.Linear(in_dim, self.hidden), nn.GELU(),
            nn.LayerNorm(self.hidden),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden, self.concept_dim),
        ).to(device)

        self._built = True

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        # A: [C, tau] or [B, C, tau]
        if A.dim() == 2:
            A = A.unsqueeze(0)
        B, C, tau = A.shape
        x = A.reshape(B, C * tau)
        self.maybe_build(C * tau, x.device)
        z = self.net(x)           # we gte the dim [B, concept_dim]
        return z


def add_adapters_(parent_module: nn.Module, args, top_level=True):
    for name, module in parent_module.named_children():
        if args.tune_mode == 'all_down_up' and isinstance(module, (nn.Conv1d, nn.Linear, transformers.Conv1D,
                                                                 nn.LayerNorm, nn.BatchNorm1d)):
            down_up.add_down_up_(parent_module, name, freeze_weight=args.freeze, merge_weights=args.merge_weights)
        elif args.tune_mode == 'down_up' and isinstance(module, (nn.Conv1d, nn.Linear, transformers.Conv1D)):
            down_up.add_down_up_(parent_module, name, freeze_weight=args.freeze, merge_weights=args.merge_weights)
        else:
            add_adapters_(module, args, False)
    return parent_module


class MechAdapt(nn.Module):

    def __init__(self, backbone, args):
        super().__init__()
        self.args = args

        if args.freeze:
            backbone.requires_grad_(False)


        def _already_has_adapters(m: nn.Module) -> bool:

            for _, sm in m.named_modules():
                if hasattr(sm, "assign_adaptation"):
                    return True
            return False


        self.backbone = backbone if _already_has_adapters(backbone) else add_adapters_(backbone, args)

        self.more_bias = not args.freeze
        self.generator = AdaptGenerator(
            backbone,
            args.concept_dim,
            activation=nn.Sigmoid if args.act == 'sigmoid' else nn.Identity,
            adaptive_dim=False,
            need_bias=self.more_bias,
            shared=not args.individual_generator,
            mid_dim=args.bottleneck_dim
        )

        #  drift encoder g(A)
        g_hidden = int(getattr(args, "cmda_g_hidden", 128))
        g_dropout = float(getattr(args, "cmda_g_dropout", 0.0))
        self.graph_encoder = GraphConceptEncoder(
            concept_dim=args.concept_dim,
            hidden=g_hidden,
            dropout=g_dropout
        )

        # store last graphs
        self.register_buffer("A_t_buf", torch.empty(0), persistent=False)
        self.register_buffer("A_ref_buf", torch.empty(0), persistent=False)


        self.ema = float(getattr(args, "cmda_graph_concept_ema", 0.0))
        if self.ema > 0:
            self.register_buffer("recent_concept", None, persistent=True)


        self.register_buffer("last_score", torch.tensor(0.0), persistent=False)

        # flags kept for compatibility with Proceed usage
        self.flag_online_learning = False
        self.flag_update = False
        self.flag_current = False
        self.flag_basic = False

        self.register_buffer("drift_buf", torch.empty(0), persistent=False)

    def _graph_drift_vector(self) -> torch.Tensor:
        if self.drift_buf.numel() == 0:
            return torch.zeros(self.args.concept_dim, device=self.drift_buf.device)

        drift = self.drift_buf
        with torch.no_grad():
            self.last_score = drift.norm(p=2).detach()
        return drift


    @torch.no_grad()
    def set_graphs(self, A_t: torch.Tensor, A_ref: typing.Optional[torch.Tensor]):
        """
     thi is called by Exp each step AFTER it computes A_t (and has access to gate.ref).
        """
        if A_t is not None:
            self.A_t_buf = A_t.detach()
        else:
            self.A_t_buf = torch.empty(0, device=self.A_t_buf.device)

        if A_ref is not None:
            self.A_ref_buf = A_ref.detach()
        else:
            self.A_ref_buf = torch.empty(0, device=self.A_ref_buf.device)

    def _graph_drift_vector(self) -> torch.Tensor:
        if self.drift_buf.numel() == 0:
            return torch.zeros(self.args.concept_dim, device=self.drift_buf.device)

        drift = self.drift_buf
        with torch.no_grad():
            self.last_score = drift.norm(p=2).detach()
        return drift

    def generate_adaptation(self) -> dict:
        drift = self._graph_drift_vector()


        if drift.dim() == 1:
            drift_in = drift.unsqueeze(0)  # [1, concept_dim]
        else:
            drift_in = drift


        if not getattr(self.args, "wo_clip", False):
            drift_in, _ = clip(drift_in)

        res = self.generator(drift_in, need_clip=False)
        # if not hasattr(self, "_dbg_every"): self._dbg_every = 200
        # if int(getattr(self, "cmda_step", 0)) % self._dbg_every == 0:
        #     # pick one bottleneck
        #     bn = next(iter(self.generator.bottlenecks.values()))
        #     wlast = bn.weights[-1]
        #     print("[BN dbg] wlast_absmax=", float(wlast.abs().max().item()),
        #           "wlast_norm=", float(wlast.norm().item()),
        #           flush=True)

        return res

    def forward(self, *x):
        if self.flag_basic:
            adaptations = {}
            for k, adapter in self.generator.bottlenecks.items():
                adaptations[k] = adapter.biases[-1] if adapter.need_bias else [None] * len(self.generator.dim_name_dict[k])
        else:
            adaptations = self.generate_adaptation()

        # assign to backbone
        for out_dim, adaptation in adaptations.items():
            for i in range(len(adaptation)):
                name = self.generator.dim_name_dict[out_dim][i]
                self.backbone.get_submodule(name).assign_adaptation(adaptation[i])
        # if not hasattr(self, "_dbg_every"):
        #     self._dbg_every = 200
        # if getattr(self, "cmda_step", 0) % self._dbg_every == 0:
        #     # pick one adaptation tensor
        #     k = next(iter(adaptations.keys()))
        #     a0 = adaptations[k][0]
        #     print("[CMDA dbg] drift_norm=", float(self.drift_buf.norm().item()) if self.drift_buf.numel() else -1,
        #           "adapt_norm=", float(a0.norm().item()) if torch.is_tensor(a0) else None)

        return self.backbone(*x)

    def encode_graph_delta(self, A_t: torch.Tensor, A_ref: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        z_t = self.graph_encoder(A_t.to(device))
        z_r = self.graph_encoder(A_ref.to(device))
        return z_t - z_r

    def freeze_adapter(self, freeze=True):
        #freeze graph_encoder + generator bottlenecks
        self.graph_encoder.requires_grad_(not freeze)
        self.graph_encoder.zero_grad(set_to_none=True)

        for adapter in self.generator.bottlenecks.values():
            adapter.weights.requires_grad_(not freeze)
            adapter.weights.zero_grad(set_to_none=True)
            adapter.biases[:len(adapter.weights) - 1].requires_grad_(not freeze)
            adapter.biases[:len(adapter.weights) - 1].zero_grad(set_to_none=True)

    def freeze_bias(self, freeze=True):
        if self.more_bias:
            for adapter in self.generator.bottlenecks.values():
                adapter.biases[-1].requires_grad_(not freeze)
                adapter.biases[-1:].zero_grad(set_to_none=True)

    def set_drift(self, drift: typing.Optional[torch.Tensor]):
        if drift is None:
            self.drift_buf = torch.empty(0, device=next(self.parameters()).device)
            return

        device = next(self.parameters()).device
        d = drift.to(device)
        if d.dim() == 2:
            d = d.squeeze(0)

        self.drift_buf = d

