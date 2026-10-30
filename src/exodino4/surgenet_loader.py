"""Strict conversion/loading of the official SurgeNetDINO DINOv2 ViT-B checkpoint."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


OFFICIAL_REPOSITORY = "https://github.com/rlpddejong/SurgeNetDINO"
OFFICIAL_WEIGHT_URL = "https://huggingface.co/rlpddejong/SurgeNetXL_DINOv1-v3/resolve/main/DINOv2_ViTb14_size336_SurgeNetXL.pth?download=true"
OFFICIAL_FILENAME = "DINOv2_ViTb14_size336_SurgeNetXL.pth"


def download_instruction(destination: str | Path) -> str:
    return f"mkdir -p {Path(destination).parent} && wget -O {destination} '{OFFICIAL_WEIGHT_URL}'"


def _unwrap(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise ValueError("SurgeNetDINO checkpoint must be a state-dict mapping.")
    for key in ("state_dict", "model", "teacher"):
        if isinstance(checkpoint.get(key), dict):
            checkpoint = checkpoint[key]
            break
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in checkpoint.items()):
        raise ValueError("Checkpoint does not contain a tensor state dict.")
    return {key.removeprefix("module.").removeprefix("model."): value for key, value in checkpoint.items()}


def _resize_position_embeddings(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Resize timm [CLS + grid] positions to the Hugging Face DINOv2 base grid."""
    if source.shape == target.shape:
        return source
    if source.ndim != 3 or target.ndim != 3 or source.shape[0] != target.shape[0] or source.shape[2] != target.shape[2]:
        raise ValueError(f"Incompatible positional embeddings: {tuple(source.shape)} -> {tuple(target.shape)}")
    source_grid = int((source.shape[1] - 1) ** 0.5)
    target_grid = int((target.shape[1] - 1) ** 0.5)
    if source_grid * source_grid + 1 != source.shape[1] or target_grid * target_grid + 1 != target.shape[1]:
        raise ValueError("Expected CLS plus square spatial positional embeddings.")
    spatial = source[:, 1:].reshape(1, source_grid, source_grid, source.shape[-1]).permute(0, 3, 1, 2)
    spatial = F.interpolate(spatial.float(), size=(target_grid, target_grid), mode="bicubic", align_corners=False)
    return torch.cat((source[:, :1], spatial.permute(0, 2, 3, 1).reshape(1, -1, source.shape[-1])), dim=1).to(source.dtype)


def _convert_timm_to_transformers(source: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], list[str]]:
    converted: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    direct = {
        "cls_token": "embeddings.cls_token", "patch_embed.proj.weight": "embeddings.patch_embeddings.projection.weight",
        "patch_embed.proj.bias": "embeddings.patch_embeddings.projection.bias", "norm.weight": "layernorm.weight", "norm.bias": "layernorm.bias",
    }
    for key, value in source.items():
        if key == "pos_embed":
            converted["embeddings.position_embeddings"] = _resize_position_embeddings(value, target["embeddings.position_embeddings"])
        elif key in direct:
            converted[direct[key]] = value
        elif key.startswith("blocks."):
            pieces = key.split(".")
            index, suffix = pieces[1], ".".join(pieces[2:])
            base = f"encoder.layer.{index}."
            mapping = {"norm1.weight": "norm1.weight", "norm1.bias": "norm1.bias", "norm2.weight": "norm2.weight", "norm2.bias": "norm2.bias",
                       "attn.proj.weight": "attention.output.dense.weight", "attn.proj.bias": "attention.output.dense.bias",
                       "mlp.fc1.weight": "mlp.fc1.weight", "mlp.fc1.bias": "mlp.fc1.bias", "mlp.fc2.weight": "mlp.fc2.weight", "mlp.fc2.bias": "mlp.fc2.bias",
                       "ls1.gamma": "layer_scale1.lambda1", "ls2.gamma": "layer_scale2.lambda1"}
            if suffix in ("attn.qkv.weight", "attn.qkv.bias"):
                destination = "attention.attention"
                for name, tensor in zip(("query", "key", "value"), value.chunk(3, dim=0)):
                    converted[f"{base}{destination}.{name}.{suffix.rsplit('.', 1)[1]}"] = tensor
            elif suffix in mapping:
                converted[base + mapping[suffix]] = value
            else:
                skipped.append(key)
        else:
            skipped.append(key)
    return converted, skipped


def load_surgenet_dinov2_vitb(backbone: torch.nn.Module, path: str | Path, source: str) -> dict[str, Any]:
    """Load the official timm ViT-B/14 checkpoint and reject incomplete backbone initialization."""
    if source != "surgenetdino_dinov2_vitb":
        raise ValueError(f"Unsupported ExoDINO4 backbone source: {source!r}")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing official SurgeNetDINO weights: {path}\nDownload with:\n{download_instruction(path)}")
    print(f"Loading SurgeNetDINO DINOv2 ViT-B checkpoint from {path}")
    source_state = _unwrap(torch.load(path, map_location="cpu", weights_only=False))
    target_state = backbone.state_dict()
    converted, skipped = _convert_timm_to_transformers(source_state, target_state)
    shape_mismatches = [key for key, value in converted.items() if key in target_state and value.shape != target_state[key].shape]
    if shape_mismatches:
        raise RuntimeError(f"Incompatible converted backbone keys: {shape_mismatches}")
    message = backbone.load_state_dict(converted, strict=False)
    missing = list(message.missing_keys)
    unexpected = list(message.unexpected_keys)
    critical_prefixes = ("embeddings.cls_token", "embeddings.position_embeddings", "embeddings.patch_embeddings.projection", "encoder.layer.", "layernorm.")
    critical_missing = [key for key in missing if key.startswith(critical_prefixes) and key != "embeddings.mask_token"]
    if critical_missing:
        raise RuntimeError(f"SurgeNetDINO initialization incomplete; critical missing keys: {critical_missing}")
    loaded_keys = [key for key in converted if key in target_state]
    loaded_parameters = sum(target_state[key].numel() for key in loaded_keys)
    total_parameters = sum(value.numel() for value in target_state.values())
    report = {"model_variant": "exodino4", "backbone_source": source, "official_repository": OFFICIAL_REPOSITORY,
              "checkpoint_path": str(path), "loaded_parameter_count": loaded_parameters, "total_backbone_parameter_count": total_parameters,
              "loaded_percentage": 100 * loaded_parameters / total_parameters, "missing_keys": missing,
              "unexpected_keys": unexpected, "skipped_source_keys": skipped, "timestamp": datetime.now(timezone.utc).isoformat()}
    print(f"Loaded {loaded_parameters}/{total_parameters} backbone parameters ({report['loaded_percentage']:.2f}%).")
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")
    print(f"Skipped documented non-backbone keys: {skipped}")
    print("SurgeNetDINO pretrained weights are CC-BY-NC-SA research-only/non-commercial.")
    return report


def save_initialization_report(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "backbone_initialization.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
