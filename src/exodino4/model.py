"""ExoDINO3-equivalent model initialized only from SurgeNetDINO DINOv2 ViT-B weights."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import Dinov2Config, Dinov2Model

from exodino3.model import SpatialGatedRefineDecoder
from .surgenet_loader import download_instruction, load_surgenet_dinov2_vitb


class ExoDINO4(nn.Module):
    """ExoDINO3 architecture with a surgical-domain SurgeNetDINO initialization."""

    diagnostic_layers = [9, 10, 11, 12]

    def __init__(self, model_name: str, num_classes: int, backbone: dict[str, Any],
                 selected_layers: list[int] | None = None, decoder_dim: int = 256,
                 dropout: float = 0.1, interpolate_pos_encoding: bool = True, **_: object) -> None:
        super().__init__()
        if backbone.get("download_if_missing", False):
            raise ValueError("ExoDINO4 never downloads weights automatically; download the official file explicitly first.")
        weights_path = Path(backbone["pretrained_weights_path"])
        if not weights_path.is_file():
            raise FileNotFoundError(f"Missing official SurgeNetDINO weights: {weights_path}\nDownload with:\n{download_instruction(weights_path)}")
        config = Dinov2Config.from_pretrained(model_name)
        self.backbone = Dinov2Model(config)
        self.initialization_report = load_surgenet_dinov2_vitb(self.backbone, backbone["pretrained_weights_path"], backbone["source"])
        self.patch_size = self.backbone.config.patch_size
        self.interpolate_pos_encoding = interpolate_pos_encoding
        self._backbone_supports_pos_interpolation = "interpolate_pos_encoding" in inspect.signature(self.backbone.forward).parameters
        if self.backbone.config.num_hidden_layers != 12:
            raise ValueError("ExoDINO4 requires a 12-block DINOv2 ViT-B backbone.")
        for parameter in self.backbone.parameters(): parameter.requires_grad = False
        for layer in self.backbone.encoder.layer[9:]:
            for parameter in layer.parameters(): parameter.requires_grad = True
        for parameter in self.backbone.layernorm.parameters(): parameter.requires_grad = True
        self.decode_head = SpatialGatedRefineDecoder(self.backbone.config.hidden_size, num_classes, decoder_dim, dropout)
        self.train(True)

    def train(self, mode: bool = True) -> "ExoDINO4":
        super().train(mode); self.backbone.eval()
        if mode:
            for layer in self.backbone.encoder.layer[9:]: layer.train(True)
            self.backbone.layernorm.train(True)
        return self

    def _tokens_to_map(self, tokens: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
        height, width = image_size; grid_h, grid_w = height // self.patch_size, width // self.patch_size
        if height % self.patch_size or width % self.patch_size: raise ValueError("Input dimensions must divide patch size.")
        expected = grid_h * grid_w
        configured = getattr(self.backbone.config, "num_register_tokens", None)
        registers = tokens.shape[1] - 1 - expected if configured is None else configured
        patch_tokens = tokens[:, 1 + registers:, :]
        assert patch_tokens.shape[1] == expected
        if (height, width) == (504, 504): assert patch_tokens.shape[1] == 1296
        return patch_tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[-1], grid_h, grid_w)

    def _backbone_outputs(self, pixel_values: torch.Tensor) -> object:
        kwargs: dict[str, object] = {"pixel_values": pixel_values, "output_hidden_states": True}
        if self.interpolate_pos_encoding and self._backbone_supports_pos_interpolation: kwargs["interpolate_pos_encoding"] = True
        return self.backbone(**kwargs)

    def extract_encoder_features(self, pixel_values: torch.Tensor) -> list[torch.Tensor]:
        outputs = self._backbone_outputs(pixel_values); size = pixel_values.shape[-2:]
        return [self._tokens_to_map(outputs.hidden_states[layer], size) for layer in (9, 10, 11)] + [self._tokens_to_map(outputs.last_hidden_state, size)]

    def forward_with_gates(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self._backbone_outputs(pixel_values); size = pixel_values.shape[-2:]
        features = [self._tokens_to_map(outputs.hidden_states[layer], size) for layer in (10, 11)] + [self._tokens_to_map(outputs.last_hidden_state, size)]
        return self.decode_head(features, size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.forward_with_gates(pixel_values)[0]
