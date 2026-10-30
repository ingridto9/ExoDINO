# ExoDINO

Clean, reproducible ExoDINO repository for the final ExoDINO3 model, the ExoDINO2 late-layer-12 comparison, the ExoDINO4 SurgNetDINO comparison, and retained Base/AB1/AB2/AB3 ablations. Model implementations are kept intact; shared refinement components are located in `src/exodino/common.py` and `src/exodino/advanced_losses.py`.

## Install

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Set `EXODISC_DATA_ROOT` to the directory containing `op1`, `op2`, `op3`, and `op4`, each with `images/` and `masks_semantic/` subdirectories:

```bash
export EXODISC_DATA_ROOT=/path/to/ExoDisc
```

The YAML files refer to this variable and contain no machine-specific dataset path. ExoDINO4 also needs the SurgNetDINO pretrained weights; set `backbone.pretrained_weights_path` in its YAML to the locally available weights file. The weights are not bundled here.

## Train

Train one ExoDINO3 fold:

```bash
python train.py --config configs/final/exodino3_fold1.yaml
```

Train folds 1 through 4 sequentially:

```bash
bash scripts/run_4fold.sh
```

Runs are written to `runs/<experiment>/fold<k>/` and include `best_val_miou.pt`, `best_val_loss.pt`, `last.pt`, `config_used.yaml`, and `history.csv`. The primary validation metric is macro mIoU over foreground classes present in the ground-truth masks; absent classes do not enter that mean.

The retained comparison and ablation configurations are under `configs/comparisons/` and `configs/legacy_ablations/`. Start them with `python train.py --config <config.yaml>`.

## Evaluate

Evaluate a fold checkpoint and write per-class IoU, Dice, precision, recall, and sensitivity to an output CSV. Classes absent from the ground truth are marked N/A; foreground macro scores include only ground-truth-present classes:

```bash
python evaluate.py --config configs/final/exodino3_fold1.yaml \
  --checkpoint artifacts/checkpoints/exodino3/fold1/best_val_miou.pt \
  --output outputs/fold1_evaluation.csv
```

## Benchmark FPS

The benchmark times the complete ExoDINO3 encoder and decoder with synthetic `[1, 3, 504, 504]` input, batch size 1, 50 warm-up iterations, and 200 timed iterations:

```bash
python benchmark_fps.py --config configs/final/exodino3_fold1.yaml \
  --checkpoint artifacts/checkpoints/exodino3/fold1/best_val_miou.pt \
  --device cuda --precision fp32 --output outputs/fold1_fps.txt
```

It reports GPU, PyTorch, CUDA, precision, mean/median latency, and FPS. Run it on the target cluster GPU for comparable measurements.

## Transfer to a cluster

The checkpoint `.pt` files stay out of ordinary Git commits. Transfer the full local folder when the weights are needed:

```bash
rsync -av --progress ExoDINO_clean/ USER@CLUSTER:/path/ExoDINO_clean/
```

Then activate the environment on the cluster, set `EXODISC_DATA_ROOT`, and use the same commands above.

## GitHub setup

From the repository root, initialize and make the first commit:

```bash
git init
git add .
git commit -m "Prepare clean ExoDINO research repository"
```

The ignore rules exclude datasets, run outputs, virtual environments, and checkpoint weights.
