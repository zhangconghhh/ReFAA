# dynamic_weights.py
from __future__ import annotations
from typing import List, Iterable, Optional, Tuple
import torch
import torch.nn as nn
from torch.autograd import grad

Tensor = torch.Tensor

# ========= 抽象基类 =========
class DynamicWeighter(nn.Module):
    """
    统一接口：
      - combine_step(losses, shared_params=None) -> (total_loss, weights_detached)
      - 对于需要 epoch 级更新的方法（如 DWA），再调用 update_epoch(avg_task_losses)
    """
    def __init__(self, num_tasks: int):
        super().__init__()
        self.num_tasks = num_tasks

    def combine_step(self, losses: List[Tensor], shared_params: Optional[Iterable[nn.Parameter]] = None
                     ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def update_epoch(self, avg_task_losses: List[float]) -> None:
        """默认不需要 epoch 级更新；DWA 会覆盖此方法。"""
        return


# ========= 1) 不确定性加权 (Kendall & Gal) =========
class UncertaintyWeighter(DynamicWeighter):
    """
    L = sum_i [ exp(-s_i) * L_i + s_i ],  其中 s_i = log(sigma_i)
    - 完全端到端：log_sigma 会与模型参数一起被优化
    - 无需 shared_params
    """
    def __init__(self, num_tasks: int, init_log_sigma: float = 0.0):
        super().__init__(num_tasks)
        self.log_sigma = nn.Parameter(torch.full((num_tasks,), float(init_log_sigma)))

    @torch.enable_grad()
    def combine_step(self, losses: List[Tensor], shared_params=None) -> Tuple[Tensor, Tensor]:
        L = torch.stack(losses)  # [T]
        w = torch.exp(-self.log_sigma)  # e^{-s}
        total = (w * L + self.log_sigma).sum()
        return total, w.detach()


# ========= 2) GradNorm (ICML'18) =========
class GradNormWeighter(DynamicWeighter):
    """
    通过让各任务在共享骨干上的梯度范数接近目标 G_hat 来自适应权重：
      - 需要传入 shared_params（共享主干的参数）
      - 仅更新本类中的权重参数（不影响模型优化器）
      - alpha ∈ [0.3, 1] 常用；默认 0.5
    """
    def __init__(
        self,
        num_tasks: int,
        shared_params: Iterable[nn.Parameter],
        alpha: float = 0.5,
        init_w: float = 1.0,
        lr: float = 1e-3
    ):
        super().__init__(num_tasks)
        self.alpha = float(alpha)
        # 用 softplus(v) 保证 w>0，数值更稳
        self.v = nn.Parameter(torch.full((num_tasks,), float(init_w)))
        self.sp = nn.Softplus()
        self._shared_params = list(shared_params)  # 只放“共享骨干”的参数！
        self._opt = torch.optim.Adam([self.v], lr=lr)
        self.register_buffer("_L0", torch.zeros(num_tasks))  # 初始损失
        self._has_L0 = False

    def _weights_pos(self) -> Tensor:
        return self.sp(self.v)

    @torch.enable_grad()
    def combine_step(self, losses: List[Tensor], shared_params=None) -> Tuple[Tensor, Tensor]:
        """
        注意：本方法会在内部计算一次与 w 相关的梯度以更新权重（开销略大）。
        然后返回当前快照权重乘以 losses 的和，用于你对模型参数的反传。
        """
        device = losses[0].device
        L = torch.stack(losses)  # [T]

        # 记录初始损失 L0（只在第一次调用时）
        if not self._has_L0:
            with torch.no_grad():
                self._L0 = L.detach()
            self._has_L0 = True

        # 当前正权重
        w = self._weights_pos()

        # --- 计算每个任务在共享参数上的梯度范数 g_i ---
        g_list = []
        for i in range(self.num_tasks):
            Li_w = w[i] * L[i]
            gi = grad(Li_w, self._shared_params, retain_graph=True, create_graph=True)
            g_norm = torch.stack([g.abs().sum() for g in gi if g is not None]).sum()
            g_list.append(g_norm)
        G = torch.stack(g_list)                 # [T]
        G_avg = G.mean()

        # 相对训练速率 r_i
        Li_by_L0 = (L / (self._L0 + 1e-12))
        ri = (Li_by_L0 / (Li_by_L0.mean() + 1e-12)).detach()
        G_hat = G_avg * (ri ** self.alpha)

        # --- 仅更新权重参数 v ---
        self._opt.zero_grad()
        w_loss = (G - G_hat).abs().sum()
        w_loss.backward()
        self._opt.step()

        # 用更新后的快照权重计算总损失（供模型参数反传）
        with torch.no_grad():
            w_now = self._weights_pos().detach()
        total = (w_now * L).sum()
        return total, w_now


# ========= 3) DWA (Dynamic Weight Averaging) =========
class DWAWeighter(DynamicWeighter):
    """
    依据 epoch 级损失变化率设权：
      lambda_i^t = softmax( (L_i^{t-1} / L_i^{t-2}) / T )
      w_i^t = T * lambda_i^t
    - 需要每个 epoch 结束时调用 update_epoch(avg_task_losses)
    - step 内直接线性加权
    """
    def __init__(self, num_tasks: int, T: float = 2.0):
        super().__init__(num_tasks)
        self.T = float(T)
        self.register_buffer("_w", torch.ones(num_tasks))
        self._hist: List[List[float]] = [[] for _ in range(num_tasks)]

    def update_epoch(self, avg_task_losses: List[float]) -> None:
        assert len(avg_task_losses) == self.num_tasks
        for i, v in enumerate(avg_task_losses):
            self._hist[i].append(float(v))

        # 历史不足2个 epoch → 等权
        if any(len(h) < 2 for h in self._hist):
            self._w = torch.ones(self.num_tasks, device=self._w.device)
            return

        ratios = torch.tensor(
            [h[-1] / (h[-2] + 1e-12) for h in self._hist],
            dtype=torch.float32, device=self._w.device
        )
        lam = torch.softmax(ratios / self.T, dim=0)  # sum=1
        self._w = self.T * lam                        # sum=T

    @torch.no_grad()
    def current_weights(self) -> Tensor:
        return self._w.detach()

    def combine_step(self, losses: List[Tensor], shared_params=None) -> Tuple[Tensor, Tensor]:
        L = torch.stack(losses)
        w = self._w.to(L.device)
        total = (w * L).sum()
        return total, w.detach()
    
    def combine_step_equal(self, losses: List[Tensor], shared_params=None) -> Tuple[Tensor, Tensor]:
        device = losses[0].device
        valid_mask = torch.tensor([l is not None for l in losses], device=device, dtype=torch.float32)
        n_valid = valid_mask.sum().clamp(min=1.0)  # 至少为1防止除零

        # 等权分配到有效项
        w = valid_mask / n_valid
        L = torch.stack([l if l is not None else torch.tensor(0., device=device) for l in losses])
        total = (w * L).sum()

        return total, w.detach()


# ========= 工厂方法 =========
def make_weighter(
    kind: str,
    num_tasks: int,
    *,
    shared_params: Optional[Iterable[nn.Parameter]] = None,
    **kwargs
) -> DynamicWeighter:
    """
    kind: 'uncertainty' | 'gradnorm' | 'dwa'
    其他超参直接通过 **kwargs 传入对应实现。
    """
    kind = kind.lower()
    if kind in ("uncertainty", "uw"):
        return UncertaintyWeighter(num_tasks=num_tasks, **kwargs)
    elif kind in ("gradnorm", "gn"):
        assert shared_params is not None, "GradNorm 需要 shared_params（共享骨干参数）"
        return GradNormWeighter(num_tasks=num_tasks, shared_params=shared_params, **kwargs)
    elif kind == "dwa":
        return DWAWeighter(num_tasks=num_tasks, **kwargs)
    else:
        raise ValueError(f"Unknown weighter kind: {kind}")
# Uncertainty：最省心，直接端到端学权重；适合任务噪声差异明显的场景。
# GradNorm：对训练动态最敏感，能实时纠偏；需要计算梯度范数，开销略大。
# DWA：开销最小，但按 epoch 级趋势更新，反应稍慢（记得每个 epoch 调 update_epoch）。