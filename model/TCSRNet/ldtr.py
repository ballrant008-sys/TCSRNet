"""LDTR: Local Discrepancy-Guided Target Reference (partial release)."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class LDTR(nn.Module):
    """Local Discrepancy-Guided Target Reference (paper Sec. III-B)."""

    def __init__(
        self,
        ch: int,
        kernels: tuple[int, ...] = (3, 5, 7),
        pool_kernel: int = 7,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not kernels:
            raise ValueError("kernels must contain at least one branch size")
        if pool_kernel % 2 == 0:
            raise ValueError("pool_kernel must be odd")

        self.kernels = tuple(kernels)
        self.pool_kernel = int(pool_kernel)
        self.eps = float(eps)

        self.context_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        ch,
                        ch,
                        kernel_size=(1, k),
                        padding=(0, k // 2),
                        groups=ch,
                        bias=False,
                    ),
                    nn.Conv2d(
                        ch,
                        ch,
                        kernel_size=(k, 1),
                        padding=(k // 2, 0),
                        groups=ch,
                        bias=False,
                    ),
                    nn.BatchNorm2d(ch),
                    nn.SiLU(inplace=True),
                )
                for k in self.kernels
            ]
        )

        self.context_selector = nn.Conv2d(3, len(self.kernels), kernel_size=1, bias=True)
        self.foreground_gate = nn.Conv2d(3, 1, kernel_size=1, bias=True)
        self.gamma = nn.Parameter(torch.full((1, ch, 1, 1), 1e-3))

    def _avg_pool(self, x: torch.Tensor) -> torch.Tensor:
        k = self.pool_kernel
        return F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)

    def _max_pool(self, x: torch.Tensor) -> torch.Tensor:
        k = self.pool_kernel
        return F.max_pool2d(x, kernel_size=k, stride=1, padding=k // 2)

    def _min_pool(self, x: torch.Tensor) -> torch.Tensor:
        return -self._max_pool(-x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Eq. (1): compact spatial response, local mean, and response residual.
        x_bar = x.mean(dim=1, keepdim=True)
        mu_t = self._avg_pool(x_bar)
        d_t = x_bar - mu_t

        # Eq. (2): local dynamic range.
        local_max = self._max_pool(x_bar)
        local_min = self._min_pool(x_bar)
        r_t = local_max - local_min

        # Eqs. (3)-(5): three complementary local contrast cues.
        c_t = F.relu(d_t) / (torch.abs(mu_t) + self.eps)
        s_t = F.relu(d_t) * torch.sigmoid(d_t / (r_t + self.eps))
        q_t = (x_bar - local_min) / (r_t + self.eps)
        cue = torch.cat([c_t, s_t, q_t], dim=1)

        # Eq. (7): contextual branch selection and soft foreground gating.
        branch_weights = torch.softmax(self.context_selector(cue), dim=1)
        foreground_mask = torch.sigmoid(self.foreground_gate(cue))

        # Eq. (8): pixel-wise aggregation of factorized depthwise branches.
        aggregated = torch.zeros_like(x)
        for idx, branch in enumerate(self.context_branches):
            aggregated = aggregated + branch(x) * branch_weights[:, idx : idx + 1]

        # Eq. (9): residual foreground modulation.
        enhanced = x + self.gamma * foreground_mask * aggregated

        # The foreground gate is retained as an explicit potential-target
        # reference and delivered to the corresponding decoder stage.
        return enhanced, foreground_mask
