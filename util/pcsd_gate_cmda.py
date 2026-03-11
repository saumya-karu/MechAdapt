# util/pcsd_gate_cmda.py

import torch
import torch.nn.functional as F


class PCSDGateCMDA:
    def __init__(
        self,
        model,
        include_keywords=("generator", "mlp1", "mlp2"),
        exclude_keywords=(),
        feat_mode="l2_l1",
        ema=0.95,
        threshold=0.15,
        warmup_steps=10,
        freeze_ref_on_fire=True,
        device=None,
    ):
        self.model = model
        self.include_keywords = include_keywords
        self.exclude_keywords = exclude_keywords
        self.feat_mode = feat_mode
        self.ema = float(ema)
        self.threshold = float(threshold)
        self.warmup_steps = int(warmup_steps)
        self.freeze_ref_on_fire = bool(freeze_ref_on_fire)
        self.device = device

        self.param_names, self.params = self._select_params_by_name(model)

        self.ref = None
        self.step = 0

    def _select_params_by_name(self, model):
        names, params = [], []
        for n, p in model.named_parameters():
            if any(k in n for k in self.exclude_keywords):
                continue
            ok = True if (self.include_keywords is None) else any(k in n for k in self.include_keywords)
            if ok:
                names.append(n)
                params.append(p)
        if len(params) == 0:
            raise RuntimeError("PCSDGateCMDA: selected zero params. Check include_keywords.")
        return names, params

    @torch.no_grad()
    def _init_ref(self, sig):
        self.ref = sig.detach().clone()

    def signature_from_loss(self, loss: torch.Tensor) -> torch.Tensor:
        device = self.device or loss.device

        orig_req = [p.requires_grad for p in self.params]
        try:
            for p in self.params:
                if not p.requires_grad:
                    p.requires_grad_(True)

            grads = torch.autograd.grad(
                loss,
                self.params,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )

            feats = []
            for g in grads:
                if g is None or (not torch.isfinite(g).all()):
                    if self.feat_mode == "l2_l1":
                        feats.extend([torch.zeros((), device=device), torch.zeros((), device=device)])
                    else:
                        feats.append(torch.zeros((), device=device))
                    continue

                g = g.detach()
                l2 = torch.norm(g.float().reshape(-1), p=2)
                if self.feat_mode == "l2_l1":
                    l1 = torch.mean(torch.abs(g.float()))
                    feats.extend([l2, l1])
                else:
                    feats.append(l2)

            sig = torch.stack(feats).to(device)
            sig = sig / (sig.norm(p=2) + 1e-12)
            return sig

        finally:
            for p, r in zip(self.params, orig_req):
                p.requires_grad_(r)

    def score(self, sig: torch.Tensor) -> torch.Tensor:
        if self.ref is None:
            return torch.tensor(0.0, device=sig.device)
        ref = self.ref.to(sig.device)
        cs = F.cosine_similarity(sig, ref, dim=0).clamp(-1.0, 1.0)
        return 1.0 - cs

    def decide(self, sig: torch.Tensor):
        self.step += 1

        if self.ref is None:
            self._init_ref(sig)
            return False, torch.tensor(0.0, device=sig.device)

        s = self.score(sig)

        if self.step <= self.warmup_steps:
            with torch.no_grad():
                self.ref = self.ema * self.ref + (1.0 - self.ema) * sig.detach()
                self.ref = self.ref / (self.ref.norm(p=2) + 1e-12)
            return False, s.detach()

        do_fire = bool(s.item() > self.threshold)


        if not (self.freeze_ref_on_fire and do_fire):
            with torch.no_grad():
                self.ref = self.ema * self.ref + (1.0 - self.ema) * sig.detach()
                self.ref = self.ref / (self.ref.norm(p=2) + 1e-12)

        return do_fire, s.detach()

