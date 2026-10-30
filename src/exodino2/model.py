"""DINOv2 final-three-block fine-tuning with a layer-12 refinement decoder."""

from __future__ import annotations

import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model

from exodino.common import DepthwiseRefinementBlock


class Layer12RefineDecoder(nn.Module):
    """Single-layer version of shared feature-space refinement decoder."""

    def __init__(self, hidden_dim: int, num_classes: int, decoder_dim: int = 256,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(hidden_dim, decoder_dim, 1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
        )
        # shared FFU, adapted only from four projected maps to one.
        self.ffu = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dim), nn.ReLU(inplace=True), nn.Dropout2d(dropout),
            nn.Conv2d(decoder_dim, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.reduce = nn.Conv2d(128, 32, 1)
        self.refinement = nn.Sequential(DepthwiseRefinementBlock(32), DepthwiseRefinementBlock(32))
        self.classifier = nn.Conv2d(32, num_classes, 1)

    def forward(self, feature: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        features = self.ffu(self.projection(feature))
        features = self.reduce(features)
        features = F.interpolate(features, size=output_size, mode="bilinear", align_corners=False)
        return self.classifier(self.refinement(features))


class ExoDINO2FT3(nn.Module):
    """Fine-tune DINOv2 blocks 10--12 and final LayerNorm; decode layer 12 only."""

    diagnostic_layers = [9, 10, 11, 12]

    def __init__(self, model_name: str, num_classes: int, selected_layers: list[int] | None = None,
                 decoder_dim: int = 256, dropout: float = 0.1,
                 interpolate_pos_encoding: bool = True, **_: object) -> None:
        super().__init__()
        self.backbone = Dinov2Model.from_pretrained(model_name)
        self.patch_size = self.backbone.config.patch_size
        self.interpolate_pos_encoding = interpolate_pos_encoding
        self._backbone_supports_pos_interpolation = "interpolate_pos_encoding" in inspect.signature(self.backbone.forward).parameters
        if self.backbone.config.num_hidden_layers != 12:
            raise ValueError("ExoDINO2-FT3 is defined for the 12-block DINOv2-Base encoder.")
        # All backbone parameters are frozen first, then only human-readable blocks 10--12
        # (zero-indexed 9--11) and the final LayerNorm are enabled.
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for layer in self.backbone.encoder.layer[9:12]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        for parameter in self.backbone.layernorm.parameters():
            parameter.requires_grad = True
        self.decode_head = Layer12RefineDecoder(self.backbone.config.hidden_size, num_classes, decoder_dim, dropout)
        self.train(True)

    def train(self, mode: bool = True) -> "ExoDINO2FT3":
        """Keep all frozen DINO components in eval mode while FT blocks stay trainable."""
        super().train(mode)
        self.backbone.eval()
        if mode:
            for layer in self.backbone.encoder.layer[9:12]:
                layer.train(True)
            self.backbone.layernorm.train(True)
        return self

    def _tokens_to_map(self, tokens: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
        """Remove CLS/register tokens and convert exactly the spatial patch tokens to a map."""
        height, width = image_size
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(f"Input {height}x{width} must be divisible by DINOv2 patch size {self.patch_size}.")
        patch_h, patch_w = height // self.patch_size, width // self.patch_size
        expected_patch_tokens = patch_h * patch_w
        # Hugging Face's Dinov2Model output has no dedicated patch-token field. Its
        # sequence is [CLS] [optional register tokens] [patch tokens]. Older
        # Transformers configs do not expose num_register_tokens, so infer it from
        # the known spatial grid in that case.
        configured_register_tokens = getattr(self.backbone.config, "num_register_tokens", None)
        num_register_tokens = (tokens.shape[1] - 1 - expected_patch_tokens
                               if configured_register_tokens is None else configured_register_tokens)
        if num_register_tokens < 0:
            raise ValueError("DINOv2 sequence is shorter than CLS plus the expected patch grid.")
        patch_tokens = tokens[:, 1 + num_register_tokens:, :]
        assert patch_tokens.shape[1] == expected_patch_tokens, (
            f"Expected {expected_patch_tokens} spatial patch tokens, got {patch_tokens.shape[1]}."
        )
        if (height, width) == (504, 504):
            assert patch_tokens.shape[1] == 1296
        return patch_tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[-1], patch_h, patch_w)

    def _backbone_outputs(self, pixel_values: torch.Tensor) -> object:
        kwargs: dict[str, object] = {"pixel_values": pixel_values, "output_hidden_states": True}
        if self.interpolate_pos_encoding and self._backbone_supports_pos_interpolation:
            kwargs["interpolate_pos_encoding"] = True
        return self.backbone(**kwargs)

    def extract_encoder_features(self, pixel_values: torch.Tensor) -> list[torch.Tensor]:
        """Return layers 9--12 to inspect the frozen-to-adapted encoder transition."""
        outputs = self._backbone_outputs(pixel_values)
        return [self._tokens_to_map(outputs.hidden_states[layer], pixel_values.shape[-2:])
                for layer in self.diagnostic_layers]

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self._backbone_outputs(pixel_values)
        # last_hidden_state is the block-12 representation after the trainable final LayerNorm.
        layer12 = self._tokens_to_map(outputs.last_hidden_state, pixel_values.shape[-2:])
        return self.decode_head(layer12, pixel_values.shape[-2:])
