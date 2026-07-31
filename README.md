# LWCamo

**LWCamo: A Lightweight Transformer-CNN Framework with Split Convolutional Attention for Camouflaged Object Detection**

LWCamo combines an ImageNet-pretrained SegFormer MiT encoder with a compact five-stage SCA decoder. This repository is the canonical, paper-consistent implementation accompanying the manuscript. Legacy notebook implementations are intentionally excluded to prevent ambiguity: the `lwcamo/` package is the single source of truth.

## Repository layout

```text
LWCamo-Code-Final/
|-- configs/                 # LWCamo-S/M/L experiment configurations
|-- datasets/                # dataset instructions and split manifests
|-- docs/                    # architecture diagrams and methodology mapping
|-- experiments/             # reproducible paper-level entry points
|-- lwcamo/
|   |-- data/                # image/mask pairing and preprocessing
|   |-- evaluation/          # standard COD metrics
|   |-- losses/              # weighted IDB loss
|   |-- models/              # SegFormer model factory and SCA block
|   `-- utils/               # deterministic execution helpers
|-- reproducibility/         # hardware record and release checklist
|-- results/                 # generated metrics, predictions, and histories
|-- scripts/                 # generic train/infer/evaluate commands
|-- tests/                   # dependency-free contract tests
`-- weights/                 # checkpoint release instructions and checksums
```

## Canonical method

- Encoder: MiT-B0, MiT-B2, or MiT-B5.
- Decoder interface: 16 x 16 x 256. B2/B5 outputs use a learned 1 x 1 projection from 512 to 256 channels.
- Decoder schedule: 256 -> 128 -> 64 -> 32 -> 16 -> 8 channels across five SCA stages.
- SCA: four channel groups, independent 3 x 3 convolutions, concatenation, **multiplicative** channel attention, then 3 x 3 stride-2 transposed convolution + BatchNorm + ReLU.
- Loss: `0.5 BCE + 2.0 Dice loss + 1.0 IoU loss`.
- Evaluation: S-alpha, adaptive E-measure, weighted F-measure, and MAE.

## Installation

```bash
conda env create -f environment.yml
conda activate lwcamo
```

Before training, audit manifests with `python scripts/check_splits.py <train> <valid>`.
Measure batch-1 latency with `python scripts/benchmark.py --model <checkpoint> --output results/benchmark.json`.

Run the dependency-free repository contract before installing the deep-learning stack:

```bash
python tests/test_repository_contract.py
```

## Data manifests

Populate `datasets/splits/` as documented in `datasets/README.md`. The standard training pool is CAMO-Train (1,000) plus COD10K-Train (3,040). CHAMELEON and NC4K are test-only.

## Training

```bash
python scripts/train.py \
  --config configs/lwcamo_small.json \
  --train-list datasets/splits/train_camo_cod10k.txt \
  --valid-list datasets/splits/valid_camo_cod10k.txt \
  --output-dir results/lwcamo_small
```

Change the configuration to `lwcamo_medium.json` or `lwcamo_large.json` for the other variants, or run `python experiments/train_all_variants.py` after populating the combined manifests.

## Inference and evaluation

```bash
python scripts/infer.py \
  --model results/lwcamo_small/best.keras \
  --images datasets/CHAMELEON/Imgs \
  --output results/predictions/CHAMELEON

python scripts/evaluate.py \
  --predictions results/predictions/CHAMELEON \
  --masks datasets/CHAMELEON/GT \
  --output results/metrics/chameleon.json
```

## Reproducibility status

Datasets, trained checkpoints, raw predictions, and hardware logs are not committed to the source repository. Populate the manifests and release evidence described in `reproducibility/CHECKLIST.md` before claiming independent reproducibility of numerical paper results. No dummy checkpoint is included.

## Methodology contract

The public implementation intentionally enforces the corrected manuscript specification:

- the decoder consumes only the deepest SegFormer feature;
- MiT-B2/B5 use a learned 1 x 1 projection from 512 to 256 channels;
- channel attention is `F = M * A`, with no residual addition;
- every SCA stage uses transposed convolution, batch normalization, and ReLU;
- CHAMELEON and NC4K are test-only datasets;
- evaluation uses unthresholded grayscale maps and the four COD metrics listed above.
