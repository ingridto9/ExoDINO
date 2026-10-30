# ExoDINO3: Gated Feature Fusion

ExoDINO3 is the final model. `ExoDINO3` adapts the last three DINOv2 blocks and fuses the spatial features from levels 10, 11, and 12 using learned per-pixel gates.

## Encoder and fine-tuning

The backbone must be 12-block DINOv2-Base. Its parameters are frozen first, then blocks 10–12 (Python indices 9–11) and the final LayerNorm are made trainable. The model's `train()` method keeps frozen modules in evaluation mode and enables training mode only for the adapted blocks and LayerNorm.

The model extracts maps from hidden states 10 and 11 and from `last_hidden_state` for level 12 after LayerNorm. Token-to-grid conversion removes CLS and register tokens. At 504×504, ViT-B/14 produces a 36×36 grid, or 1,296 patch tokens.

## Fusion and decoder

Each map passes through a 1×1 projection, BatchNorm, and ReLU. The three projected maps are concatenated and passed to the gate head: a 1×1 convolution, GroupNorm, ReLU, and a three-channel output. Softmax across channels produces three weights at each spatial position. The features are summed using those weights, allowing the contribution from each level to vary across the image.

The fused feature map passes through the convolutional decoder (including dropout), is reduced to 32 channels, resized to the input resolution, refined by two depthwise blocks, and classified per pixel. `forward()` returns logits; `forward_with_gates()` also returns the gates for analysis.

## Configurations

The four canonical folds are in `../../configs/final/`. `selected_layers` remains in the YAML for compatibility, but ExoDINO3's fusion is defined by the model and always uses levels 10–12.
