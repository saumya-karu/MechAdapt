
# util/graph_drift_gate.py

from __future__ import annotations
import torch
import torch.nn.functional as F


class GraphDriftGate:
    def __init__(
        self,
        threshold: float = 0.15,
        ema: float = 0.98,
        warmup: int = 50,
        stable_thresh: float = 0.08,
        stable_patience: int = 40,


        y_index: int | None = None,       #f set, compare only A[y_index, :]
        beta_l2: float = 0.10,            #weight on L2 term
        normalize: str = "row",           #"row", "global", "none"
        freeze_ref_during_warmup: bool = True,
        update_ref_when_stable: bool = True,
        eps: float = 1e-12,
    ):
        self.threshold = float(threshold)
        self.ema = float(ema)
        self.warmup = int(warmup)

        self.stable_thresh = float(stable_thresh)
        self.stable_patience = int(stable_patience)

        # new
        self.y_index = y_index if (y_index is None) else int(y_index)
        self.beta_l2 = float(beta_l2)
        self.normalize = str(normalize).lower()
        self.freeze_ref_during_warmup = bool(freeze_ref_during_warmup)
        self.update_ref_when_stable = bool(update_ref_when_stable)
        self.eps = float(eps)

        self.ref = None
        self.step = 0
        self.stable_count = 0


        self._warmup_sum = None
        self._warmup_n = 0


    def _normalize_A(self, A: torch.Tensor) -> torch.Tensor:
        if self.normalize == "none":
            return A
        if self.normalize == "global":
            return A / (A.norm(p=2) + self.eps)
        if self.normalize == "row":

            row_norm = A.norm(p=2, dim=-1, keepdim=True).clamp_min(self.eps)
            return A / row_norm
        raise ValueError(f"normalize must be one of ['row','global','none'], got {self.normalize}")

    def _select_view(self, A: torch.Tensor) -> torch.Tensor:

        if self.y_index is not None:
            yi = self.y_index
            yi = max(-A.size(0), min(A.size(0) - 1, yi))
            v = A[yi, :]            # [tau]
            return v.reshape(-1)
        return A.reshape(-1)


    @torch.no_grad()
    def _set_ref(self, A: torch.Tensor):
        A = A.detach()
        A = self._normalize_A(A)
        self.ref = A.clone()

    @torch.no_grad()
    def _ema_update_ref(self, A: torch.Tensor):
        A = A.detach()
        A = self._normalize_A(A)
        if self.ref is None:
            self.ref = A.clone()
        else:
            self.ref = self.ema * self.ref + (1.0 - self.ema) * A


    def score(self, A: torch.Tensor) -> torch.Tensor:
        if self.ref is None:
            return torch.tensor(0.0, device=A.device)

        A_n = self._normalize_A(A)
        R_n = self.ref.to(A.device)

        a = self._select_view(A_n)
        r = self._select_view(R_n)

        #cosine distance
        cs = F.cosine_similarity(a, r, dim=0).clamp(-1.0, 1.0)
        d_cos = 1.0 - cs

        #magnitude change
        d_l2 = torch.norm(a - r, p=2)

        return d_cos + self.beta_l2 * d_l2


    def decide(self, A: torch.Tensor):

        self.step += 1
        A = A.detach()

        if self.ref is None:
            # start warmup accumulation
            A0 = self._normalize_A(A)
            self._warmup_sum = A0.clone()
            self._warmup_n = 1
            self.ref = A0.clone()
            return False, torch.tensor(0.0, device=A.device)

        if self.step <= self.warmup:
            s = self.score(A)

            if self.freeze_ref_during_warmup:

                A0 = self._normalize_A(A)
                if self._warmup_sum is None:
                    self._warmup_sum = A0.clone()
                    self._warmup_n = 1
                else:
                    self._warmup_sum = self._warmup_sum + A0
                    self._warmup_n += 1


                if self.step == self.warmup and self._warmup_sum is not None:
                    mean_A = self._warmup_sum / max(self._warmup_n, 1)
                    self._set_ref(mean_A)
            else:

                self._ema_update_ref(A)

            return False, s.detach()


        s = self.score(A)


        if float(s.item()) < self.stable_thresh:
            self.stable_count += 1
        else:
            self.stable_count = 0

        do_fire = bool(float(s.item()) > self.threshold)

        if self.update_ref_when_stable and (not do_fire) and (self.stable_count >= self.stable_patience):
            self._ema_update_ref(A)

        return do_fire, s.detach()


