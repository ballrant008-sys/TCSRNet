from pathlib import Path

import torch.nn as nn
import yaml


class TCSRNet(nn.Module):
    def __init__(self, config_path="configs/tcsrnet.yaml", backend=None):
        super().__init__()
        with Path(config_path).open("r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f) or {}

        model_cfg = self.cfg.get("model", {})
        self.in_channels = model_cfg.get("in_channels", 1)
        self.out_channels = model_cfg.get("out_channels", 1)
        self.input_size = model_cfg.get("input_size", 256)

        if backend is not None and not isinstance(backend, nn.Module):
            raise TypeError("backend must be an nn.Module")
        self.backend = backend

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError("Expected input with shape [B, C, H, W]")
        if self.backend is None:
            raise RuntimeError("Full model backend is not included in this partial release")
        return self.backend(x)
