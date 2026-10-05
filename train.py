"""Training entry point for one ExoDisc leave-one-operation-out fold."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchmetrics.classification as tc
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from exodino.data import ExoDiscSemanticDataset, build_train_transform
from exodino.losses import SegmentationLoss
from exodino.model import DinoSurgicalSegmenter


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


def run_epoch(model: torch.nn.Module, loader: DataLoader, criterion: SegmentationLoss, optimizer: torch.optim.Optimizer | None, scaler: torch.amp.GradScaler, device: torch.device, num_classes: int, amp_enabled: bool) -> tuple[float, float, float, list[float]]:
    """Run one epoch using the same macro/micro IoU metrics as the notebook."""
    is_training = optimizer is not None
    model.train(is_training)
    macro_iou = tc.MulticlassJaccardIndex(num_classes=num_classes, ignore_index=0, average="macro").to(device)
    micro_iou = tc.MulticlassJaccardIndex(num_classes=num_classes, ignore_index=0, average="micro").to(device)
    class_iou = tc.MulticlassJaccardIndex(num_classes=num_classes, ignore_index=0, average="none").to(device)
    total_loss = 0.0

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

            if is_training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()
            predictions = logits.argmax(dim=1)
            macro_iou.update(predictions, masks)
            micro_iou.update(predictions, masks)
            class_iou.update(predictions, masks)

    return total_loss / len(loader), macro_iou.compute().item(), micro_iou.compute().item(), class_iou.compute().detach().cpu().tolist()


def build_dataset(config: dict[str, Any], operation_ids: list[int], is_training: bool) -> ExoDiscSemanticDataset:
    """Create a dataset with augmentation only for the training split."""
    data_cfg = config["data"]
    augmentation_cfg = config["augmentation"]
    return ExoDiscSemanticDataset(
        root_dir=data_cfg["root_dir"],
        operation_ids=operation_ids,
        image_size=data_cfg["image_size"],
        num_classes=config["model"]["num_classes"],
        operation_prefix=data_cfg["operation_prefix"],
        images_dir_name=data_cfg["images_dir_name"],
        masks_dir_name=data_cfg["masks_dir_name"],
        transform=build_train_transform(data_cfg["image_size"]) if is_training and augmentation_cfg["enabled"] else None,
        cutmix_probability=augmentation_cfg["cutmix_probability"] if is_training else 0.0,
        cutmix_alpha=augmentation_cfg["cutmix_alpha"],
        include_empty_masks=data_cfg["include_empty_masks"],
    )


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
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    train_dataset = build_dataset(config, data_cfg["train_operations"], is_training=True)
    val_dataset = build_dataset(config, data_cfg["val_operations"], is_training=False)
    loader_kwargs = {"batch_size": data_cfg["batch_size"], "num_workers": data_cfg["num_workers"], "pin_memory": torch.cuda.is_available()}
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = training_cfg["amp"] and device.type == "cuda"
    model = DinoSurgicalSegmenter(**model_cfg).to(device)
    # Compute class frequencies without random augmentation or CutMix.
    weight_dataset = build_dataset(config, data_cfg["train_operations"], is_training=False)
    class_weights = compute_class_weights(weight_dataset, model_cfg["num_classes"]).to(device)
    criterion = SegmentationLoss(class_weights, training_cfg["dice_weight"], training_cfg["ce_weight"])

    trainable_backbone = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": trainable_backbone, "lr": training_cfg["lr_backbone"]},
        {"params": model.decode_head.parameters(), "lr": training_cfg["lr_head"]},
    ], weight_decay=training_cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_val_loss, history, start_epoch = float("inf"), [], 1
    if training_cfg["resume"]:
        checkpoint = torch.load(training_cfg["resume"], map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        best_val_loss = checkpoint["best_val_loss"]
        history = checkpoint.get("history", [])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, training_cfg["epochs"] + 1):
        train_loss, train_miou, train_miou_micro, train_iou = run_epoch(model, train_loader, criterion, optimizer, scaler, device, model_cfg["num_classes"], amp_enabled)
        val_loss, val_miou, val_miou_micro, val_iou = run_epoch(model, val_loader, criterion, None, scaler, device, model_cfg["num_classes"], amp_enabled)
        row: dict[str, Any] = {"epoch": epoch, "train_loss": train_loss, "train_miou": train_miou, "train_miou_micro": train_miou_micro, "val_loss": val_loss, "val_miou": val_miou, "val_miou_micro": val_miou_micro}
        row.update({f"train_iou_{name}": score for name, score in zip(data_cfg["class_names"], train_iou)})
        row.update({f"val_iou_{name}": score for name, score in zip(data_cfg["class_names"], val_iou)})
        history.append(row)
        save_history(history, output_dir / "history.csv")

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scaler_state_dict": scaler.state_dict(), "best_val_loss": best_val_loss, "config": config, "history": history}
        torch.save(checkpoint, output_dir / "last.pt")
        if is_best:
            torch.save(checkpoint, output_dir / "best.pt")

        print(f"Epoch {epoch:03d} | train loss {train_loss:.4f}, macro/micro mIoU {train_miou:.4f}/{train_miou_micro:.4f} | val loss {val_loss:.4f}, macro/micro mIoU {val_miou:.4f}/{val_miou_micro:.4f}")


if __name__ == "__main__":
    main()
