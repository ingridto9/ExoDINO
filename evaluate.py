"""Evaluate one ExoDINO checkpoint on its configured held-out fold."""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from exodino.data import ExoDiscSemanticDataset
from exodino.model import DinoSurgicalSegmenter
from exodino2.model import ExoDINO2FT3
from exodino3.data import ExoDINO3Dataset
from exodino3.model import ExoDINO3
from exodino4.model import ExoDINO4


def model_from_config(config: dict[str, Any]) -> torch.nn.Module:
    model_cfg = config["model"]
    variant = model_cfg.get("model_variant", "exodino")
    kwargs = {key: value for key, value in model_cfg.items() if key != "model_variant"}
    if variant == "exodino":
        return DinoSurgicalSegmenter(**kwargs)
    if variant == "exodino2_ft3":
        return ExoDINO2FT3(**kwargs)
    if variant == "exodino3":
        return ExoDINO3(**kwargs)
    if variant == "exodino4":
        return ExoDINO4(**kwargs, backbone=config["backbone"])
    raise ValueError(f"Unsupported model variant: {variant}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation.csv"))
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data, model_cfg = config["data"], config["model"]
    data["root_dir"] = os.path.expandvars(data["root_dir"])
    if "$" in data["root_dir"]:
        raise ValueError("Set EXODISC_DATA_ROOT or configure data.root_dir.")
    num_classes = model_cfg["num_classes"]
    if len(data["class_names"]) != num_classes:
        raise ValueError("data.class_names and model.num_classes must have the same length.")

    dataset_class = ExoDINO3Dataset if model_cfg.get("model_variant") in {"exodino3", "exodino4"} else ExoDiscSemanticDataset
    dataset_kwargs: dict[str, Any] = {
        "root_dir": data["root_dir"],
        "operation_ids": data["val_operations"],
        "image_size": data["image_size"],
        "num_classes": num_classes,
        "operation_prefix": data["operation_prefix"],
        "images_dir_name": data["images_dir_name"],
        "masks_dir_name": data["masks_dir_name"],
        "transform": None,
        "cutmix_probability": 0.0,
        "include_empty_masks": True,
    }
    if dataset_class is ExoDINO3Dataset:
        dataset_kwargs["overlay_mask"] = config.get("overlay_mask")
    dataset = dataset_class(**dataset_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=data["num_workers"],
        pin_memory=device.type == "cuda",
    )
    model = model_from_config(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].long().to(device, non_blocking=True)
            prediction = model(images).argmax(dim=1)
            encoded = (target * num_classes + prediction).reshape(-1).cpu()
            confusion += torch.bincount(encoded, minlength=num_classes**2).reshape(num_classes, num_classes)

    true_positive = confusion.diag().double()
    gt_pixels = confusion.sum(dim=1).double()
    predicted_pixels = confusion.sum(dim=0).double()
    false_positive = predicted_pixels - true_positive
    false_negative = gt_pixels - true_positive
    ground_truth_present = gt_pixels > 0

    # A class absent from the ground truth is N/A for these per-class measures.
    iou = torch.full((num_classes,), float("nan"), dtype=torch.double)
    dice = torch.full_like(iou, float("nan"))
    precision = torch.full_like(iou, float("nan"))
    recall = torch.full_like(iou, float("nan"))
    iou[ground_truth_present] = true_positive[ground_truth_present] / (
        gt_pixels[ground_truth_present] + predicted_pixels[ground_truth_present] - true_positive[ground_truth_present]
    ).clamp_min(1)
    dice[ground_truth_present] = 2 * true_positive[ground_truth_present] / (
        2 * true_positive[ground_truth_present] + false_positive[ground_truth_present] + false_negative[ground_truth_present]
    ).clamp_min(1)
    precision[ground_truth_present] = true_positive[ground_truth_present] / predicted_pixels[ground_truth_present].clamp_min(1)
    recall[ground_truth_present] = true_positive[ground_truth_present] / gt_pixels[ground_truth_present].clamp_min(1)
    sensitivity = recall.clone()

    macro_classes = ground_truth_present.clone()
    macro_classes[0] = False  # Preserve the project's foreground-only macro convention.
    metrics = {"iou": iou, "dice": dice, "precision": precision, "recall": recall, "sensitivity": sensitivity}
    macro = {
        name: float(values[macro_classes].mean()) if macro_classes.any() else math.nan
        for name, values in metrics.items()
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["class_id", "class_name", "ground_truth_present", "gt_pixels", "predicted_pixels",
               "tp", "fp", "fn", "iou", "dice", "precision", "recall", "sensitivity", "n_macro_classes"]
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns)
        writer.writeheader()
        for class_id, class_name in enumerate(data["class_names"]):
            present = bool(ground_truth_present[class_id])
            writer.writerow({
                "class_id": class_id,
                "class_name": class_name,
                "ground_truth_present": present,
                "gt_pixels": int(gt_pixels[class_id]),
                "predicted_pixels": int(predicted_pixels[class_id]),
                "tp": int(true_positive[class_id]),
                "fp": int(false_positive[class_id]),
                "fn": int(false_negative[class_id]),
                **{name: f"{values[class_id].item():.8f}" if present else "" for name, values in metrics.items()},
                "n_macro_classes": "",
            })
        writer.writerow({
            "class_name": "macro_foreground_gt_present",
            **{name: f"{value:.8f}" if math.isfinite(value) else "" for name, value in macro.items()},
            "n_macro_classes": int(macro_classes.sum()),
        })

    print(f"Wrote {args.output}")
    print("Foreground macro over classes present in ground truth: " + ", ".join(
        f"{name}={value:.6f}" for name, value in macro.items()
    ))


if __name__ == "__main__":
    main()
