"""Shared lightweight convolution and sampling utilities."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        s: int = 1,
        p: int | None = None,
        groups: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=k,
            stride=s,
            padding=p,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class OffsetSampler(nn.Module):
    """Bilinear sampler using offsets represented in pixel units."""

    def __init__(self, align_corners: bool = True) -> None:
        super().__init__()
        self.align_corners = bool(align_corners)

    def forward(self, x: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        if offset.ndim != 4 or offset.shape[1] != 2:
            raise ValueError("offset must have shape [B, 2, H, W]")

        b, _, h, w = x.shape
        if offset.shape[-2:] != (h, w):
            raise ValueError("offset spatial size must match sampled feature size")

        if self.align_corners:
            ys = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype)
            xs = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype)
            norm_x = 2.0 / max(w - 1, 1)
            norm_y = 2.0 / max(h - 1, 1)
        else:
            ys = (torch.arange(h, device=x.device, dtype=x.dtype) + 0.5) * 2.0 / h - 1.0
            xs = (torch.arange(w, device=x.device, dtype=x.dtype) + 0.5) * 2.0 / w - 1.0
            norm_x = 2.0 / max(w, 1)
            norm_y = 2.0 / max(h, 1)

        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
        normalized_offset = torch.stack(
            [offset[:, 0] * norm_x, offset[:, 1] * norm_y],
            dim=-1,
        )
        sampling_grid = base_grid + normalized_offset
        return F.grid_sample(
            x,
            sampling_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=self.align_corners,
        )
