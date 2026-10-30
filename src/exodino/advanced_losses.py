"""Selective foreground loss shared by the retained fine-tuned DINOv2 models."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def capped_inverse_sqrt_class_weights(pixel_counts: torch.Tensor, epsilon: float = 1e-8,
                                      max_weight: float = 5.0) -> torch.Tensor:
    """Compute mean-one capped inverse-square-root weights from fold pixel counts."""
    frequencies = pixel_counts.to(torch.float64) / pixel_counts.sum()
    weights = (frequencies + epsilon).rsqrt().clamp(max=max_weight)
    return (weights / weights.mean()).float()


class SelectiveForegroundLoss(nn.Module):
    """Balanced CE plus a foreground Tversky term that weights false positives more."""

    def __init__(self, class_weights: torch.Tensor, lambda_tversky: float = 1.0,
                 alpha: float = 0.7, beta: float = 0.3, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.register_buffer("class_weights", class_weights.float())
        self.lambda_tversky, self.alpha, self.beta, self.epsilon = lambda_tversky, alpha, beta, epsilon
        self.last_components: dict[str, torch.Tensor] = {}

    def components(self, logits: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        balanced_ce = F.cross_entropy(logits, target, weight=self.class_weights)
        probabilities = logits.softmax(dim=1)
        one_hot = F.one_hot(target, num_classes=logits.shape[1]).permute(0, 3, 1, 2).to(probabilities.dtype)
        foreground_probability, foreground_target = probabilities[:, 1:], one_hot[:, 1:]
        dimensions = (0, 2, 3)
        tp = (foreground_probability * foreground_target).sum(dim=dimensions)
        fp = (foreground_probability * (1 - foreground_target)).sum(dim=dimensions)
        fn = ((1 - foreground_probability) * foreground_target).sum(dim=dimensions)
        tversky = (tp + self.epsilon) / (tp + self.alpha * fp + self.beta * fn + self.epsilon)
        foreground_tversky = 1 - tversky.mean()
        total = balanced_ce + self.lambda_tversky * foreground_tversky
        return {"total": total, "balanced_ce": balanced_ce, "foreground_tversky": foreground_tversky}

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.last_components = self.components(logits, target)
        return self.last_components["total"]
