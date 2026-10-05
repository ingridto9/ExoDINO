import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor, dice_weight: float = 1.0, ce_weight: float = 1.0, smooth: float = 1e-6):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_weight, self.ce_weight, self.smooth = dice_weight, ce_weight, smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = self.cross_entropy(logits, target)
        probabilities = logits.softmax(dim=1)
        one_hot = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2).float()
        intersection = (probabilities * one_hot).sum(dim=(2, 3))
        cardinality = (probabilities + one_hot).sum(dim=(2, 3))
        dice_loss = 1 - ((2 * intersection + self.smooth) / (cardinality + self.smooth)).mean()
        return self.dice_weight * dice_loss + self.ce_weight * ce_loss
