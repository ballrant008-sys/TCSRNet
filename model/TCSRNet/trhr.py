"""TRHR: Target-Response Hierarchical Rectification."""
from __future__ import annotations
from typing import Iterable
import torch
from torch import nn
import torch.nn.functional as F
from .blocks import ConvBNAct, OffsetSampler

class TRHR(nn.Module):
    """Target-Response Hierarchical Rectification module."""

    def __init__(
        self,
        ch: int,
        eta: float = 2.0,
        ref_points: Iterable[tuple[float, float]] | None = None,
        residual_mode: str = "signed",
    ) -> None:
        super().__init__()
        if ref_points is None:
            ref_points = (
                (0.0, 0.0),
                (-1.0, 0.0),
                (1.0, 0.0),
                (0.0, -1.0),
                (0.0, 1.0),
            )
        self.ref_points = tuple((float(x), float(y)) for x, y in ref_points)
        if not self.ref_points:
            raise ValueError("ref_points must not be empty")

        self.num_reference_points = len(self.ref_points)
        self.eta = float(eta)
        if residual_mode not in ("signed", "legacy_abs"):
            raise ValueError("residual_mode must be signed or legacy_abs")
        self.residual_mode = residual_mode

        self.current_projection = ConvBNAct(ch, ch, k=1)
        self.semantic_projection = ConvBNAct(ch, ch, k=1)

        # Eq. (trhr_2): Conv3x3 -> Conv1x1 -> sigmoid.
        self.target_response_head = nn.Sequential(
            ConvBNAct(ch, ch, k=3),
            nn.Conv2d(ch, 1, kernel_size=1, bias=True),
        )
        # Learnable strength for injecting the LDTR reference into the
        # current-scale target-response logit. It starts from zero so the
        # explicit path is introduced without destabilizing initialization.
        self.reference_strength = nn.Parameter(torch.tensor(0.0))

        # Eq. (trhr_3): independent 2D offsets for all K reference points.
        self.offset_head = nn.Conv2d(
            ch,
            2 * self.num_reference_points,
            kernel_size=3,
            padding=1,
            bias=True,
        )

        # Learnable aggregation weights w_q in Eq. (trhr_4).
        self.reference_weight_logits = nn.Parameter(torch.zeros(self.num_reference_points))
        self.sampler = OffsetSampler(align_corners=True)

        # Eq. (trhr_6): joint reconstruction of A_i, S_hat_i, and Z_i.
        self.reconstruction = ConvBNAct(ch * 3, ch, k=3)

    def forward(
        self,
        boundary_enhanced_feature: torch.Tensor,
        deeper_rectified_feature: torch.Tensor,
        target_reference: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Eq. (trhr_1).
        current = self.current_projection(boundary_enhanced_feature)
        semantic = self.semantic_projection(deeper_rectified_feature)
        semantic = F.interpolate(
            semantic,
            size=current.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # Eqs. (trhr_2)-(trhr_3), explicitly conditioned by the LDTR
        # potential-target reference at the same scale.
        target_logit = self.target_response_head(current)
        # None means the reference is disabled, not a strongly negative
        # reference logit. This matters for w/o-LDTR ablations.
        if target_reference is not None:
            if target_reference.shape[-2:] != target_logit.shape[-2:]:
                target_reference = F.interpolate(
                    target_reference,
                    size=target_logit.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            target_reference = target_reference.clamp(1e-4, 1.0 - 1e-4)
            target_logit = target_logit + self.reference_strength * torch.logit(target_reference)
        target_response = torch.sigmoid(target_logit)
        b, _, h, w = current.shape
        initial_offsets = torch.tanh(self.offset_head(current)).view(
            b,
            self.num_reference_points,
            2,
            h,
            w,
        )
        constrained_offsets = self.eta * target_response.unsqueeze(1) * initial_offsets

        # Eq. (trhr_4): weighted local-reference bilinear sampling.
        weights = torch.softmax(self.reference_weight_logits, dim=0)
        rectified_semantic = torch.zeros_like(semantic)
        for idx, (ref_x, ref_y) in enumerate(self.ref_points):
            reference = torch.zeros_like(constrained_offsets[:, idx])
            reference[:, 0] = ref_x
            reference[:, 1] = ref_y
            sampled = self.sampler(semantic, constrained_offsets[:, idx] + reference)
            rectified_semantic = rectified_semantic + weights[idx] * sampled

        # Eqs. (trhr_5)-(trhr_6).
        rectification_residual = current - rectified_semantic
        if self.residual_mode == "legacy_abs":
            rectification_residual = rectification_residual.abs()
        return self.reconstruction(
            torch.cat([current, rectified_semantic, rectification_residual], dim=1)
        )
