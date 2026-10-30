"""Training entry point for one ExoDisc leave-one-operation-out fold."""

from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from exodino.data import ExoDiscSemanticDataset, build_train_transform
from exodino.losses import SegmentationLoss
from exodino.model import DinoSurgicalSegmenter
from exodino.advanced_losses import SelectiveForegroundLoss, capped_inverse_sqrt_class_weights
from exodino2.model import ExoDINO2FT3
from exodino3.data import ExoDINO3Dataset
from exodino3.model import ExoDINO3
from exodino4.model import ExoDINO4
from exodino4.surgenet_loader import save_initialization_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ExoDINO on one leave-one-operation-out fold.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a YAML experiment configuration.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Make data ordering and augmentation sampling reproducible where possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(dataset: ExoDiscSemanticDataset, num_classes: int) -> torch.Tensor:
    """Compute weights from the original training masks before any augmentation."""
    pixel_counts = torch.zeros(num_classes, dtype=torch.float64)
    for mask in tqdm(dataset.iter_raw_masks(), total=len(dataset), desc="Computing class weights"):
        pixel_counts += torch.bincount(torch.from_numpy(mask).long().flatten(), minlength=num_classes)

    if (pixel_counts == 0).any():
        missing = torch.where(pixel_counts == 0)[0].tolist()
        raise ValueError(f"Classes without pixels in this training fold: {missing}")

    weights = pixel_counts.sum() / (num_classes * pixel_counts)
    return (weights / weights.mean()).float()


def compute_capped_class_weights(dataset: ExoDiscSemanticDataset, num_classes: int) -> torch.Tensor:
    """Estimate capped weights from unaugmented masks in this fold."""
    pixel_counts = torch.zeros(num_classes, dtype=torch.float64)
    for mask in tqdm(dataset.iter_raw_masks(), total=len(dataset), desc="Computing balanced class weights"):
        pixel_counts += torch.bincount(torch.from_numpy(mask).long().flatten(), minlength=num_classes)
    return capped_inverse_sqrt_class_weights(pixel_counts)


def run_epoch(model: torch.nn.Module, loader: DataLoader, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer | None, scaler: torch.amp.GradScaler, device: torch.device, num_classes: int, amp_enabled: bool) -> tuple[float, float, float, list[float], dict[str, float]]:
    """Run one epoch using the same macro/micro IoU metrics as the notebook."""
    is_training = optimizer is not None
    model.train(is_training)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    total_loss = 0.0
    component_sums: dict[str, float] = {}
    presence_batches = torch.zeros(12, dtype=torch.long); presence_total = 0; background_batches = 0

    gradient_context = torch.enable_grad() if is_training else torch.no_grad()
    with gradient_context:
        for batch in tqdm(loader, desc="Train" if is_training else "Validation", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            if is_training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, masks)
            if isinstance(criterion, SelectiveForegroundLoss):
                for name, value in criterion.last_components.items():
                    component_sums[name] = component_sums.get(name, 0.0) + value.detach().item()

            if is_training:
                scaler.scale(loss).backward()
                if hasattr(model, "assert_frozen_backbone_grads"):
                    model.assert_frozen_backbone_grads()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()
            predictions = logits.argmax(dim=1)
            encoded = masks.reshape(-1).long() * num_classes + predictions.reshape(-1).long()
            confusion += torch.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)

    true_pixels = confusion.sum(dim=1).float()
    predicted_pixels = confusion.sum(dim=0).float()
    true_positive = confusion.diag().float()
    union = true_pixels + predicted_pixels - true_positive
    scores = torch.full((num_classes,), float("nan"), device=device)
    present = true_pixels > 0
    scores[present] = true_positive[present] / union[present].clamp_min(1)
    foreground_present = present[1:]
    macro_value = scores[1:][foreground_present].mean().item() if foreground_present.any() else float("nan")
    foreground_union = union[1:][foreground_present].sum()
    micro_value = (true_positive[1:][foreground_present].sum() / foreground_union.clamp_min(1)).item() if foreground_present.any() else float("nan")
    return (total_loss / len(loader), macro_value, micro_value,
            scores.detach().cpu().tolist(),
            {**{name: value / len(loader) for name, value in component_sums.items()},
             })


def build_dataset(config: dict[str, Any], operation_ids: list[int], is_training: bool) -> ExoDiscSemanticDataset:
    """Create a dataset with augmentation only for the training split."""
    data_cfg = config["data"]
    data_cfg["root_dir"] = os.path.expandvars(data_cfg["root_dir"])
    if "$" in data_cfg["root_dir"]:
        raise ValueError("Set EXODISC_DATA_ROOT or configure data.root_dir before training.")
    augmentation_cfg = config["augmentation"]
    dataset_class = ExoDINO3Dataset if config["model"].get("model_variant") in {"exodino3", "exodino4"} else ExoDiscSemanticDataset
    dataset_kwargs: dict[str, Any] = {
        "root_dir": data_cfg["root_dir"], "operation_ids": operation_ids, "image_size": data_cfg["image_size"],
        "num_classes": config["model"]["num_classes"], "operation_prefix": data_cfg["operation_prefix"],
        "images_dir_name": data_cfg["images_dir_name"], "masks_dir_name": data_cfg["masks_dir_name"],
        "transform": build_train_transform(data_cfg["image_size"]) if is_training and augmentation_cfg["enabled"] else None,
        "cutmix_probability": augmentation_cfg["cutmix_probability"] if is_training else 0.0,
        "cutmix_alpha": augmentation_cfg["cutmix_alpha"], "include_empty_masks": data_cfg["include_empty_masks"],
    }
    if dataset_class is ExoDINO3Dataset:
        dataset_kwargs["overlay_mask"] = config.get("overlay_mask")
    return dataset_class(**dataset_kwargs)


def save_history(history: list[dict[str, Any]], output_path: Path) -> None:
    """Write all epoch metrics to a human-readable CSV file."""
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    set_seed(config["seed"])

    data_cfg = config["data"]
    model_cfg = config["model"]
    training_cfg = config["training"]
    if data_cfg["image_size"] % 14:
        raise ValueError("data.image_size must be divisible by 14 for the ViT-B/14 patch grid.")
    if len(data_cfg["class_names"]) != model_cfg["num_classes"]:
        raise ValueError("data.class_names and model.num_classes must have the same length.")
    if set(data_cfg["train_operations"]) & set(data_cfg["val_operations"]):
        raise ValueError("Training and validation operations must not overlap.")

    output_dir = Path(config["output_dir"]) / config["experiment_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_used.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    train_dataset = build_dataset(config, data_cfg["train_operations"], is_training=True)
    val_dataset = build_dataset(config, data_cfg["val_operations"], is_training=False)
    loader_kwargs = {"batch_size": data_cfg["batch_size"], "num_workers": data_cfg["num_workers"], "pin_memory": torch.cuda.is_available()}
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = training_cfg["amp"] and device.type == "cuda"
    model_variant = model_cfg.get("model_variant", "exodino")
    model_kwargs = {key: value for key, value in model_cfg.items() if key != "model_variant"}
    if model_variant == "exodino":
        model = DinoSurgicalSegmenter(**model_kwargs).to(device)
    elif model_variant == "exodino2_ft3":
        model = ExoDINO2FT3(**model_kwargs).to(device)
    elif model_variant == "exodino3":
        model = ExoDINO3(**model_kwargs).to(device)
    elif model_variant == "exodino4":
        model = ExoDINO4(**model_kwargs, backbone=config["backbone"]).to(device)
        save_initialization_report(model.initialization_report, output_dir)
    else:
        raise ValueError(f"Unsupported model.model_variant {model_variant!r}.")
    # Compute class frequencies without random augmentation or CutMix.
    weight_dataset = build_dataset(config, data_cfg["train_operations"], is_training=False)
    if model_variant in {"exodino2_ft3", "exodino3", "exodino4"}:
        class_weights = compute_capped_class_weights(weight_dataset, model_cfg["num_classes"]).to(device)
        print(f"{model_variant} capped inverse-sqrt class weights (mean=1): "
              + ", ".join(f"{name}={weight:.6f}" for name, weight in zip(data_cfg["class_names"], class_weights.tolist())))
        criterion: torch.nn.Module = SelectiveForegroundLoss(class_weights, lambda_tversky=1.0)
        if model_variant in {"exodino2_ft3", "exodino3", "exodino4"}:
            backbone_trainable = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
            trainable_names = [name for name, parameter in model.backbone.named_parameters() if parameter.requires_grad]
            print(f"{model_variant} parameters: trainable={sum(p.numel() for p in model.parameters() if p.requires_grad)}, "
                  f"frozen={sum(p.numel() for p in model.parameters() if not p.requires_grad)}")
            print(f"{model_variant} trainable DINO parameters: " + ", ".join(trainable_names))
            print(f"{model_variant} confirmation: only Transformer blocks 10-12 and final LayerNorm are trainable.")
            optimizer = torch.optim.AdamW([
                {"params": backbone_trainable, "lr": training_cfg["lr_backbone"]},
                {"params": model.decode_head.parameters(), "lr": training_cfg["lr_head"]},
            ], weight_decay=training_cfg["weight_decay"])
        else:
            optimizer = torch.optim.AdamW(model.decode_head.parameters(), lr=training_cfg["lr_head"],
                                          weight_decay=training_cfg["weight_decay"])
    else:
        class_weights = compute_class_weights(weight_dataset, model_cfg["num_classes"]).to(device)
        criterion = SegmentationLoss(class_weights, training_cfg["dice_weight"], training_cfg["ce_weight"])
        trainable_backbone = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW([
            {"params": trainable_backbone, "lr": training_cfg["lr_backbone"]},
            {"params": model.decode_head.parameters(), "lr": training_cfg["lr_head"]},
        ], weight_decay=training_cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_val_loss, best_val_miou, history, start_epoch = float("inf"), float("-inf"), [], 1
    if training_cfg["resume"]:
        checkpoint = torch.load(training_cfg["resume"], map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        best_val_loss = checkpoint["best_val_loss"]
        best_val_miou = checkpoint.get("best_val_miou", float("-inf"))
        history = checkpoint.get("history", [])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, training_cfg["epochs"] + 1):
        train_loss, train_miou, train_miou_micro, train_iou, train_components = run_epoch(model, train_loader, criterion, optimizer, scaler, device, model_cfg["num_classes"], amp_enabled)
        val_loss, val_miou, val_miou_micro, val_iou, val_components = run_epoch(model, val_loader, criterion, None, scaler, device, model_cfg["num_classes"], amp_enabled)
        row: dict[str, Any] = {"epoch": epoch, "train_loss": train_loss, "train_miou": train_miou, "train_miou_micro": train_miou_micro, "val_loss": val_loss, "val_miou": val_miou, "val_miou_micro": val_miou_micro}
        row.update({f"train_iou_{name}": score for name, score in zip(data_cfg["class_names"], train_iou)})
        row.update({f"val_iou_{name}": score for name, score in zip(data_cfg["class_names"], val_iou)})
        row.update({f"train_{name}": value for name, value in train_components.items()})
        row.update({f"val_{name}": value for name, value in val_components.items()})
        history.append(row)
        save_history(history, output_dir / "history.csv")
        is_best_loss = val_loss < best_val_loss
        is_best_miou = val_miou > best_val_miou
        if is_best_loss:
            best_val_loss = val_loss
        if is_best_miou:
            best_val_miou = val_miou
        checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                      "scaler_state_dict": scaler.state_dict(), "best_val_loss": best_val_loss,
                      "best_val_miou": best_val_miou, "config": config, "history": history}
        checkpoint.update({"val_loss": val_loss, "val_miou": val_miou, "train_loss": train_loss,
                           "train_miou": train_miou, "scheduler_state_dict": None})
        torch.save(checkpoint, output_dir / "last.pt")
        if is_best_loss:
            torch.save(checkpoint, output_dir / "best_val_loss.pt")
        if is_best_miou:
            torch.save(checkpoint, output_dir / "best_val_miou.pt")

        print(f"Epoch {epoch:03d} | train loss {train_loss:.4f}, macro/micro mIoU {train_miou:.4f}/{train_miou_micro:.4f} | val loss {val_loss:.4f}, macro/micro mIoU {val_miou:.4f}/{val_miou_micro:.4f}")



if __name__ == "__main__":
    main()
