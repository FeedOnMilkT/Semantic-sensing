"""Multi-task loss = per-task losses + a weighting scheme (CLAUDE.md §5).

Schemes (cfg.loss.weighting.name):
  - fixed:       static per-task weights (default seg 1.0, depth 0.5, normal 0.5)
  - uncertainty: Kendall 2018 homoscedastic uncertainty — learnable log_var per
                 task; total = Σ 0.5*exp(-s_t)*L_t + 0.5*s_t. log_vars are params,
                 so MultiTaskLoss must be passed to the optimizer.
  - dwa:         Dynamic Weight Averaging — weight ∝ softmax of the per-task loss
                 descent rate over the last two steps (Liu 2019).

MultiTaskLoss only computes losses for tasks present in BOTH the model output and
the batch targets, so single-dataset / conditional tasks work automatically.
"""
import torch
import torch.nn as nn

from .task_losses import build_task_loss

_DEFAULT_W = {"seg": 1.0, "depth": 0.5, "normal": 0.5}


class MultiTaskLoss(nn.Module):
    def __init__(self, tasks, task_loss_cfgs=None, weighting=None):
        super().__init__()
        task_loss_cfgs = task_loss_cfgs or {}
        self.tasks = list(tasks)
        self.losses = nn.ModuleDict({t: build_task_loss(t, task_loss_cfgs.get(t)) for t in self.tasks})

        weighting = dict(weighting or {"name": "fixed"})
        self.scheme = weighting.pop("name", "fixed")
        if self.scheme == "uncertainty":
            self.log_vars = nn.ParameterDict(
                {t: nn.Parameter(torch.zeros(())) for t in self.tasks})
        elif self.scheme == "fixed":
            w = weighting.get("weights", {})
            self.fixed_w = {t: float(w.get(t, _DEFAULT_W.get(t, 1.0))) for t in self.tasks}
        elif self.scheme == "dwa":
            self.dwa_T = float(weighting.get("temperature", 2.0))
            self.register_buffer("_prev1", torch.ones(len(self.tasks)))
            self.register_buffer("_prev2", torch.ones(len(self.tasks)))
            self._dwa_ready = False
        else:
            raise KeyError(f"unknown weighting scheme: {self.scheme}")

    def _dwa_weights(self, device):
        if not self._dwa_ready:
            return torch.ones(len(self.tasks), device=device)
        r = (self._prev1 / self._prev2.clamp(min=1e-8)).to(device)
        w = torch.softmax(r / self.dwa_T, dim=0) * len(self.tasks)
        return w

    def forward(self, preds, targets, tasks):
        active = [t for t in tasks if t in self.losses and t in preds and t in targets]
        raw = {t: self.losses[t](preds[t], targets[t]) for t in active}
        device = next(iter(raw.values())).device if raw else torch.device("cpu")

        if self.scheme == "uncertainty":
            total = sum(0.5 * torch.exp(-self.log_vars[t]) * raw[t] + 0.5 * self.log_vars[t] for t in active)
        elif self.scheme == "fixed":
            total = sum(self.fixed_w[t] * raw[t] for t in active)
        else:  # dwa
            w = self._dwa_weights(device)
            idx = {t: i for i, t in enumerate(self.tasks)}
            total = sum(w[idx[t]] * raw[t] for t in active)
            # shift history with this step's losses
            cur = self._prev1.clone()
            for t in active:
                cur[idx[t]] = float(raw[t].detach())
            self._prev2 = self._prev1.clone()
            self._prev1 = cur
            self._dwa_ready = True

        return total, {t: float(raw[t].detach()) for t in active}


def build_loss(cfg, tasks):
    return MultiTaskLoss(
        tasks=tasks,
        task_loss_cfgs=cfg.get("task_losses", {}) if cfg else {},
        weighting=cfg.get("weighting", {"name": "fixed"}) if cfg else {"name": "fixed"},
    )
