"""Dataset extension that masks the ORBEYE overlay before augmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from exodino.data import ExoDiscSemanticDataset, _read_semantic_mask


class ExoDINO3Dataset(ExoDiscSemanticDataset):
    """ExoDisc dataset with an optional RGB-only, pre-augmentation overlay mask."""

    def __init__(self, *args: Any, overlay_mask: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.overlay_mask = overlay_mask or {"enabled": False}
        self._validate_overlay_mask()

    def _validate_overlay_mask(self) -> None:
        if not self.overlay_mask.get("enabled", False):
            return
        if self.overlay_mask.get("fill") != "imagenet_mean":
            raise ValueError("ExoDINO3 overlay_mask.fill must be 'imagenet_mean'.")
        coordinates = [self.overlay_mask.get(key) for key in ("x0_fraction", "y0_fraction", "x1_fraction", "y1_fraction")]
        if any(value is None or not 0 <= float(value) <= 1 for value in coordinates):
            raise ValueError("overlay mask fractions must be between 0 and 1.")
        if coordinates[0] >= coordinates[2] or coordinates[1] >= coordinates[3]:
            raise ValueError("overlay mask requires x0 < x1 and y0 < y1.")

    def _mask_overlay(self, image: np.ndarray) -> np.ndarray:
        """Mask relative coordinates in raw-scale RGB before resize or augmentation."""
        if not self.overlay_mask.get("enabled", False):
            return image
        height, width = image.shape[:2]
        x0 = int(np.floor(float(self.overlay_mask["x0_fraction"]) * width))
        x1 = int(np.ceil(float(self.overlay_mask["x1_fraction"]) * width))
        y0 = int(np.floor(float(self.overlay_mask["y0_fraction"]) * height))
        y1 = int(np.ceil(float(self.overlay_mask["y1_fraction"]) * height))
        masked = image.copy()
        # ImageNet RGB mean in the source image's 0--255 scale.
        masked[y0:y1, x0:x1] = np.array([0.485, 0.456, 0.406]) * 255.0
        return masked

    def _load_raw_rgb_and_mask(self, index: int, apply_overlay_mask: bool) -> tuple[np.ndarray, np.ndarray, str]:
        image_path, mask_path = self.samples[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if apply_overlay_mask:
            image = self._mask_overlay(image)
        mask = _read_semantic_mask(mask_path, self.num_classes)
        return image, mask, image_path.name

    def load_unmasked_rgb(self, index: int) -> np.ndarray:
        """Return the validation RGB at model resolution, without overlay masking."""
        image, _, _ = self._load_raw_rgb_and_mask(index, apply_overlay_mask=False)
        return cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

    def _load_sample(self, index: int) -> tuple[np.ndarray, np.ndarray, str]:
        image, mask, image_name = self._load_raw_rgb_and_mask(index, apply_overlay_mask=True)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
        return image, mask, image_name
