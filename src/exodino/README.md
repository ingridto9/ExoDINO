# ExoDINO Base and Initial Ablations

This package implements the multi-level DINOv2 model used as the base and for the Base, AB1, AB2, and AB3 ablations. The main model is `DinoSurgicalSegmenter`; its decoder is `DinoFeatureFusionDecoder`.

## Model flow

1. The input is an RGB batch `[B, 3, H, W]`. Height and width must be divisible by the DINOv2 patch size, which is 14 for ViT-B/14.
2. `Dinov2Model.from_pretrained(model_name)` extracts the hidden states. `selected_layers` chooses which intermediate maps to use. These are Hugging Face hidden-state indices, where index 0 is the embedding output before the first Transformer block.
3. For every selected layer, the model removes the CLS token and reshapes the patch tokens into a 2D grid. A 1×1 convolution, BatchNorm, and ReLU project each map to `decoder_dim` channels.
4. The maps are resized to the deepest map's grid size and concatenated. The decoder produces class logits, which are then interpolated to the original image resolution.

The default decoder (`decoder_variant: refinement`) uses 3×3 convolutions, BatchNorm, ReLU, and dropout before classification. AB3 sets `decoder_variant: linear`: after concatenation, it uses only a 1×1 convolution to predict the classes.

## Fine-tuning

`freeze_backbone` freezes the encoder. If it is enabled and `unfreeze_last_n_layers` is greater than zero, the specified number of final blocks and the final LayerNorm are made trainable. The decoder remains trainable.

## Configurations

Reproducible configurations are in `../../configs/legacy_ablations/base/`, `../../configs/legacy_ablations/AB1_layer12/`, `../../configs/legacy_ablations/AB2_frozen/`, and `../../configs/legacy_ablations/AB3_linear_decoder/`. The main settings are `model.selected_layers`, `model.freeze_backbone`, `model.unfreeze_last_n_layers`, and `model.decoder_variant`.

## Shared components

`common.py` contains the depthwise refinement block reused by ExoDINO2 and ExoDINO3. `advanced_losses.py` contains the foreground loss and class weights used by the fine-tuned variants.
