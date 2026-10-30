# ExoDINO4 / SurgNetDINO

ExoDINO4 keeps ExoDINO3's fusion and decoder architecture. The comparison therefore isolates backbone initialization: it uses SurgeNetDINO surgical-domain DINOv2 ViT-B weights instead of generic DINOv2 weights.

## Backbone initialization

The model builds `Dinov2Model` from the DINOv2-Base configuration and loads the surgical weights with `load_surgenet_dinov2_vitb`. The loader returns an initialization report, which the training script saves. Automatic downloads are disabled: `backbone.pretrained_weights_path` must point to the official local weight file, and `backbone.source` identifies the supported format. If the file is missing, model construction stops with an explicit error.

After initialization, the encoder weights are frozen and blocks 10–12 and the final LayerNorm are made trainable, as in ExoDINO3. The forward pass, patch-token handling, and gated fusion are described in `../exodino3/README.md`.

## Configurations

Comparison folds are in `../../configs/comparisons/exodino4_surgnetdino/`. Before training, set the path to the accessible SurgeNetDINO checkpoint in the configuration. The weights are not included in this repository.
