# ExoDINO2: Layer-12-Only Decoder

This package implements `ExoDINO2FT3`, the comparison model that decodes only the final DINOv2 representation instead of fusing multiple levels as ExoDINO3 does.

## Encoder and fine-tuning

The backbone is 12-block DINOv2-Base, loaded with `Dinov2Model.from_pretrained`. All weights are frozen first; Transformer blocks 10–12 (Python indices 9–11) and the final LayerNorm are then made trainable. Frozen modules remain in evaluation mode during training.

The forward pass requests hidden states and uses `last_hidden_state`, the output of block 12 after the final LayerNorm. The code removes CLS and any register tokens, then reshapes the patch tokens into a spatial grid. The `selected_layers` argument is accepted for compatibility with shared YAML files, but this variant always decodes layer 12 only.

## Decoder

`Layer12RefineDecoder` projects the map with a 1×1 convolution, then applies a feature refinement unit with 3×3 convolutions and dropout. A projection reduces the features to 32 channels; the map is resized to the input resolution, refined by two `DepthwiseRefinementBlock`s, and classified with a 1×1 convolution. The output is a per-pixel class-logit map.

## Configurations

Use the files in `../../configs/comparisons/exodino2_layer12_only/`. `selected_layers: [12]` documents the variant; the implementation determines the selected representation.
