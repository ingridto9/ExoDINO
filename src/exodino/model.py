from __future__ import annotations

import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model


class DinoFeatureFusionDecoder(nn.Module):
    """Project and fuse feature maps extracted from several DINOv2 blocks."""
    def __init__(self, hidden_dim: int, num_classes: int, decoder_dim: int = 256, dropout: float = 0.1, num_features: int = 4):
        super().__init__()
        self.projections = nn.ModuleList([nn.Sequential(nn.Conv2d(hidden_dim, decoder_dim, 1, bias=False), nn.BatchNorm2d(decoder_dim), nn.ReLU(inplace=True)) for _ in range(num_features)])
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * num_features, decoder_dim, 3, padding=1, bias=False), nn.BatchNorm2d(decoder_dim), nn.ReLU(inplace=True), nn.Dropout2d(dropout),
            nn.Conv2d(decoder_dim, decoder_dim // 2, 3, padding=1, bias=False), nn.BatchNorm2d(decoder_dim // 2), nn.ReLU(inplace=True),
            nn.Conv2d(decoder_dim // 2, num_classes, 1),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        target_size = features[-1].shape[-2:]
        projected = []
        for feature, projection in zip(features, self.projections):
            feature = projection(feature)
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)
            projected.append(feature)
        return self.fuse(torch.cat(projected, dim=1))


class DinoSurgicalSegmenter(nn.Module):
    """DINOv2 encoder with a lightweight multi-level semantic decoder."""
    def __init__(self, model_name: str, num_classes: int, selected_layers: list[int], freeze_backbone: bool = True, unfreeze_last_n_layers: int = 1, decoder_dim: int = 256, dropout: float = 0.1, interpolate_pos_encoding: bool = True):
        super().__init__()
        self.backbone = Dinov2Model.from_pretrained(model_name)
        self.patch_size = self.backbone.config.patch_size
        self.selected_layers = selected_layers
        self.interpolate_pos_encoding = interpolate_pos_encoding
        self._backbone_supports_pos_interpolation = "interpolate_pos_encoding" in inspect.signature(self.backbone.forward).parameters
        if not all(1 <= layer <= self.backbone.config.num_hidden_layers for layer in selected_layers):
            raise ValueError("selected_layers must refer to Transformer blocks, starting from 1.")
        self._configure_backbone(freeze_backbone, unfreeze_last_n_layers)
        self.decode_head = DinoFeatureFusionDecoder(self.backbone.config.hidden_size, num_classes, decoder_dim, dropout, len(selected_layers))

    def _configure_backbone(self, freeze_backbone: bool, unfreeze_last_n_layers: int) -> None:
        """Freeze the encoder, optionally keeping its final blocks trainable."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = not freeze_backbone
        if freeze_backbone and unfreeze_last_n_layers > 0:
            for layer in self.backbone.encoder.layer[-unfreeze_last_n_layers:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True
            for parameter in self.backbone.layernorm.parameters():
                parameter.requires_grad = True

    def _tokens_to_map(self, tokens: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
        """Convert DINO patch tokens to a 2D feature map for convolutional decoding."""
        height, width = image_size
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(f"Input {height}x{width} must be divisible by DINOv2 patch size {self.patch_size}.")
        patches_h, patches_w = height // self.patch_size, width // self.patch_size
        patch_tokens = tokens[:, 1:, :]
        if patch_tokens.shape[1] != patches_h * patches_w:
            raise ValueError("DINOv2 token grid does not match the input size.")
        return patch_tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[-1], patches_h, patches_w)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Match the original notebook when the installed Transformers version exposes
        # this argument; the signature check keeps the package compatible otherwise.
        backbone_kwargs = {"pixel_values": pixel_values, "output_hidden_states": True}
        if self.interpolate_pos_encoding and self._backbone_supports_pos_interpolation:
            backbone_kwargs["interpolate_pos_encoding"] = True
        outputs = self.backbone(**backbone_kwargs)
        features = [self._tokens_to_map(outputs.hidden_states[layer], pixel_values.shape[-2:]) for layer in self.selected_layers]
        return F.interpolate(self.decode_head(features), size=pixel_values.shape[-2:], mode="bilinear", align_corners=False)
