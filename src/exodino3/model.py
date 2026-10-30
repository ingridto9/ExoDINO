"""Final-three-block DINOv2 adaptation with spatial gated feature fusion."""

from __future__ import annotations

import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model

from exodino.common import DepthwiseRefinementBlock


class SpatialGatedRefineDecoder(nn.Module):
    """Fuse projected layers 10--12 by per-pixel softmax gates before ExoDINO2's FFU."""

    def __init__(self, hidden_dim: int, num_classes: int, decoder_dim: int = 256,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Conv2d(hidden_dim, decoder_dim, 1, bias=False),
                          nn.BatchNorm2d(decoder_dim), nn.ReLU(inplace=True))
            for _ in range(3)
        ])
        self.gate_head = nn.Sequential(
            nn.Conv2d(decoder_dim * 3, 64, 1, bias=False), nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 1),
        )
        self.ffu = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dim), nn.ReLU(inplace=True), nn.Dropout2d(dropout),
            nn.Conv2d(decoder_dim, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.reduce = nn.Conv2d(128, 32, 1)
        self.refinement = nn.Sequential(DepthwiseRefinementBlock(32), DepthwiseRefinementBlock(32))
        self.classifier = nn.Conv2d(32, num_classes, 1)

    def forward(self, features: list[torch.Tensor], output_size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        projected = [projection(feature) for feature, projection in zip(features, self.projections)]
        gates = self.gate_head(torch.cat(projected, dim=1)).softmax(dim=1)
        fused = sum(gates[:, index:index + 1] * feature for index, feature in enumerate(projected))
        assert fused.shape[1:] == (256, features[0].shape[-2], features[0].shape[-1])
        ffu_features = self.ffu(fused)
        high_resolution = F.interpolate(self.reduce(ffu_features), size=output_size, mode="bilinear", align_corners=False)
        return self.classifier(self.refinement(high_resolution)), gates


class ExoDINO3(nn.Module):
    """Adapt blocks 10--12 and gate their feature maps for full-resolution segmentation."""

    diagnostic_layers = [9, 10, 11, 12]
    decoder_layers = [10, 11, 12]

    def __init__(self, model_name: str, num_classes: int, selected_layers: list[int] | None = None,
                 decoder_dim: int = 256, dropout: float = 0.1,
                 interpolate_pos_encoding: bool = True, **_: object) -> None:
        super().__init__()
        self.backbone = Dinov2Model.from_pretrained(model_name)
        self.patch_size = self.backbone.config.patch_size
        self.interpolate_pos_encoding = interpolate_pos_encoding
        self._backbone_supports_pos_interpolation = "interpolate_pos_encoding" in inspect.signature(self.backbone.forward).parameters
        if self.backbone.config.num_hidden_layers != 12:
            raise ValueError("ExoDINO3 is defined for 12-block DINOv2-Base.")
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for layer in self.backbone.encoder.layer[9:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        for parameter in self.backbone.layernorm.parameters():
            parameter.requires_grad = True
        self.decode_head = SpatialGatedRefineDecoder(self.backbone.config.hidden_size, num_classes, decoder_dim, dropout)
        self.train(True)

    def train(self, mode: bool = True) -> "ExoDINO3":
        """Restore frozen/trainable DINO module modes after the parent mode transition."""
        super().train(mode)
        self.backbone.eval()
        if mode:
            for layer in self.backbone.encoder.layer[9:]:
                layer.train(True)
            self.backbone.layernorm.train(True)
        return self

    def _tokens_to_map(self, tokens: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
        height, width = image_size
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(f"Input {height}x{width} must be divisible by patch size {self.patch_size}.")
        grid_h, grid_w = height // self.patch_size, width // self.patch_size
        expected = grid_h * grid_w
        configured_registers = getattr(self.backbone.config, "num_register_tokens", None)
        registers = tokens.shape[1] - 1 - expected if configured_registers is None else configured_registers
        if registers < 0:
            raise ValueError("DINOv2 sequence is shorter than CLS plus expected spatial tokens.")
        patch_tokens = tokens[:, 1 + registers:, :]
        assert patch_tokens.shape[1] == expected
        if (height, width) == (504, 504):
            assert patch_tokens.shape[1] == 1296
        return patch_tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[-1], grid_h, grid_w)

    def _backbone_outputs(self, pixel_values: torch.Tensor) -> object:
        kwargs: dict[str, object] = {"pixel_values": pixel_values, "output_hidden_states": True}
        if self.interpolate_pos_encoding and self._backbone_supports_pos_interpolation:
            kwargs["interpolate_pos_encoding"] = True
        return self.backbone(**kwargs)

    def _decoder_features(self, outputs: object, image_size: tuple[int, int]) -> list[torch.Tensor]:
        # Use post-LayerNorm output for F12 so the final DINO LayerNorm is trained.
        return [self._tokens_to_map(outputs.hidden_states[layer], image_size) for layer in (10, 11)] + [
            self._tokens_to_map(outputs.last_hidden_state, image_size)
        ]

    def extract_encoder_features(self, pixel_values: torch.Tensor) -> list[torch.Tensor]:
        """Return diagnostic maps for frozen layer 9 and adapted layers 10--12."""
        outputs = self._backbone_outputs(pixel_values)
        return [self._tokens_to_map(outputs.hidden_states[layer], pixel_values.shape[-2:]) for layer in (9, 10, 11)] + [
            self._tokens_to_map(outputs.last_hidden_state, pixel_values.shape[-2:])
        ]

    def forward_with_gates(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self._backbone_outputs(pixel_values)
        return self.decode_head(self._decoder_features(outputs, pixel_values.shape[-2:]), pixel_values.shape[-2:])

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.forward_with_gates(pixel_values)[0]
