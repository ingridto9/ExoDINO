"""Dataset utilities for the released ExoDisc semantic-segmentation masks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, Sequence

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# The DINOv2 encoder expects ImageNet-normalised RGB tensors.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def build_train_transform(image_size: int) -> A.Compose:
    """Return the augmentation pipeline used only for training images."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.80, 1.20), translate_percent={"x": (-0.08, 0.08), "y": (-0.08, 0.08)}, rotate=(-15, 15), shear={"x": (-3, 3), "y": (-3, 3)}, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, keep_ratio=True, fit_output=False, rotate_method="largest_box", balanced_scale=True, p=0.75),
        A.Perspective(scale=(0.001, 0.003), keep_size=True, fit_output=False, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.15),
        A.CropNonEmptyMaskIfExists(height=int(image_size * 0.85), width=int(image_size * 0.85), ignore_values=[0], p=0.35),
        A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
        A.CLAHE(clip_limit=(1.0, 2.0), tile_grid_size=(8, 8), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.RandomGamma(gamma_limit=(70, 130), p=0.4),
        A.HueSaturationValue(hue_shift_limit=(-5, 5), sat_shift_limit=(-10, 10), val_shift_limit=(-10, 10), p=0.35),
        A.OneOf([A.GaussNoise(std_range=(0.05, 0.10), p=0.6), A.ISONoise(color_shift=(0.01, 0.04), intensity=(0.1, 0.6), p=0.4)], p=0.6),
        A.Sharpen(alpha=(0.1, 0.5), lightness=(0.5, 1.0), p=0.3),
    ])


class ExoDiscSemanticDataset(Dataset):
    """Load RGB images and their single-channel semantic PNG masks.

    Released mask IDs: 0 background, 1 aspirator, 2 burr, 3 retractor,
    4 spatula, 5 forceps, 6 scalpel, 7 curettes, 8 electrocautery, 9 dura,
    10 ligament, 11 herniation, and 12 disc.
    """

    def __init__(self, root_dir: str | Path, operation_ids: Sequence[int], image_size: int, num_classes: int, operation_prefix: str = "OP", images_dir_name: str = "images", masks_dir_name: str = "semantic_masks", transform: A.Compose | None = None, cutmix_probability: float = 0.0, cutmix_alpha: float = 1.0, include_empty_masks: bool = True) -> None:
        self.image_size, self.num_classes = image_size, num_classes
        self.transform = transform
        self.cutmix_probability, self.cutmix_alpha = cutmix_probability, cutmix_alpha
        self.samples: list[tuple[Path, Path]] = []
        root = Path(root_dir)

        for operation_id in operation_ids:
            operation_dir = root / f"{operation_prefix}{operation_id}"
            image_dir, mask_dir = operation_dir / images_dir_name, operation_dir / masks_dir_name
            if not image_dir.is_dir() or not mask_dir.is_dir():
                raise FileNotFoundError(f"Expected '{images_dir_name}' and '{masks_dir_name}' inside {operation_dir}. Set their names in the YAML configuration if required.")
            image_paths = sorted((path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES), key=lambda path: _natural_key(path.name))
            for image_path in image_paths:
                # Images and masks are matched by filename stem, for example frame_0001.png.
                mask_path = mask_dir / f"{image_path.stem}.png"
                if not mask_path.is_file():
                    raise FileNotFoundError(f"Missing semantic mask for {image_path.name}: {mask_path}")
                if include_empty_masks or self._mask_has_foreground(mask_path):
                    self.samples.append((image_path, mask_path))
        if not self.samples:
            raise RuntimeError("No image-mask pairs were found.")

    def __len__(self) -> int:
        return len(self.samples)

    def iter_raw_masks(self) -> Iterator[np.ndarray]:
        """Yield original-resolution masks without resize, augmentation, or CutMix.

        This method is used for class-frequency estimation so that training weights
        always describe the released annotations rather than synthetic samples.
        """
        for _, mask_path in self.samples:
            yield _read_semantic_mask(mask_path, self.num_classes)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image, mask, image_name = self._load_sample(index)
        if self.cutmix_probability > 0 and np.random.random() < self.cutmix_probability:
            other_image, other_mask, _ = self._load_sample(np.random.randint(len(self)))
            image, mask = _cutmix(image, mask, other_image, other_mask, self.cutmix_alpha)
        image_tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).long()
        return {"image": (image_tensor - IMAGENET_MEAN) / IMAGENET_STD, "mask": mask_tensor, "image_name": image_name}

    def _load_sample(self, index: int) -> tuple[np.ndarray, np.ndarray, str]:
        image_path, mask_path = self.samples[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = _read_semantic_mask(mask_path, self.num_classes)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
        return image, mask, image_path.name

    @staticmethod
    def _mask_has_foreground(mask_path: Path) -> bool:
        return bool(np.any(_read_semantic_mask(mask_path, num_classes=13) != 0))


def _read_semantic_mask(mask_path: Path, num_classes: int) -> np.ndarray:
    """Read a class-ID PNG and reject colour masks or invalid class IDs."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Could not read semantic mask: {mask_path}")
    if mask.ndim != 2:
        raise ValueError(f"Semantic mask must be single-channel, got shape {mask.shape}: {mask_path}")
    if mask.min() < 0 or mask.max() >= num_classes:
        raise ValueError(f"Mask {mask_path} contains invalid class IDs: {np.unique(mask).tolist()}")
    return mask.astype(np.uint8, copy=False)


def _cutmix(image_a: np.ndarray, mask_a: np.ndarray, image_b: np.ndarray, mask_b: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Replace the same random rectangle in an image and semantic mask."""
    height, width = image_a.shape[:2]
    lam = np.random.beta(alpha, alpha)
    cut_width, cut_height = int(width * np.sqrt(1 - lam)), int(height * np.sqrt(1 - lam))
    center_x, center_y = np.random.randint(width), np.random.randint(height)
    x1, x2 = max(0, center_x - cut_width // 2), min(width, center_x + cut_width // 2)
    y1, y2 = max(0, center_y - cut_height // 2), min(height, center_y + cut_height // 2)
    image_a[y1:y2, x1:x2] = image_b[y1:y2, x1:x2]
    mask_a[y1:y2, x1:x2] = mask_b[y1:y2, x1:x2]
    return image_a, mask_a


def _natural_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]
