"""LGBI: Low-Frequency-Guided Boundary Interaction."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from .blocks import ConvBNAct

class HaarLowFrequency(nn.Module):
    """One-level Haar DWT followed by LL-only inverse reconstruction.

    For orthonormal Haar, retaining only LL coefficients reproduces each
    2x2 block as its mean. This implementation uses the mathematically
    equivalent parameter-free block-average and nearest reconstruction.
    Odd spatial extents use replicate padding on the bottom/right.
    This differs from the original package's 5x5 stride-one average pooling.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        pad_h, pad_w = h % 2, w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        ll_only = F.avg_pool2d(x, kernel_size=2, stride=2)
        return F.interpolate(ll_only, scale_factor=2, mode="nearest")[..., :h, :w]

class LGBI(nn.Module):
    """Low-Frequency-Guided Boundary Interaction module."""

    def __init__(self, ch: int, pool_kernel: int = 5, low_frequency_mode: str = "haar") -> None:
        super().__init__()
        if low_frequency_mode not in ("haar", "legacy_avg"):
            raise ValueError("low_frequency_mode must be haar or legacy_avg")
        self.low_frequency_mode = low_frequency_mode
        self.haar_low_frequency = HaarLowFrequency()
        if pool_kernel % 2 == 0:
            raise ValueError("pool_kernel must be odd")
        self.pool_kernel = int(pool_kernel)

        self.feature_projection = ConvBNAct(ch * 2, ch, k=1)
        self.context_gate = nn.Sequential(
            nn.Conv2d(ch + 1, ch, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.boundary_head = nn.Sequential(
            ConvBNAct(ch, ch, k=3),
            nn.Conv2d(ch, 1, kernel_size=1, bias=True),
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.feature_fusion = ConvBNAct(ch * 2, ch, k=3)

    def _avg_pool(self, x: torch.Tensor) -> torch.Tensor:
        k = self.pool_kernel
        return F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)

    def forward(
        self,
        encoder_feature: torch.Tensor,
        deeper_rectified_feature: torch.Tensor,
        target_reference: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Eq. (lgbi_up) and Eq. (lgbi_1).
        upsampled = F.interpolate(
            deeper_rectified_feature,
            size=encoder_feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        hidden = self.feature_projection(torch.cat([encoder_feature, upsampled], dim=1))

        # Eq. (lgbi_2): local low-frequency trend and residual response.
        low_frequency = (self.haar_low_frequency(hidden) if self.low_frequency_mode == "haar"
                         else self._avg_pool(hidden))
        residual = hidden - low_frequency

        # Explicit LDTR-reference-conditioned contextual regulation.
        if target_reference is None:
            target_reference = torch.zeros_like(low_frequency[:, :1])
        elif target_reference.shape[-2:] != low_frequency.shape[-2:]:
            target_reference = F.interpolate(
                target_reference,
                size=low_frequency.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        target_reference = target_reference.clamp(0.0, 1.0)
        contextual_weight = self.context_gate(
            torch.cat([low_frequency, target_reference], dim=1)
        )
        regulated_residual = residual * contextual_weight

        # Eq. (lgbi_4): boundary response prediction.
        boundary_logit = self.boundary_head(regulated_residual)
        boundary_probability = torch.sigmoid(boundary_logit)

        # Eq. (lgbi_7): boundary-guided encoder refinement.
        refined_encoder = encoder_feature * (1.0 + self.alpha * boundary_probability)

        # Eq. (lgbi_8): boundary-enhanced feature for TRHR.
        boundary_enhanced = self.feature_fusion(torch.cat([refined_encoder, upsampled], dim=1))
        return boundary_enhanced, boundary_probability
