# Dataset organization

Datasets are intentionally excluded from version control. Arrange them as:

```text
datasets/
  CAMO/Imgs  CAMO/GT
  COD10K/Imgs  COD10K/GT
  CHAMELEON/Imgs  CHAMELEON/GT
  NC4K/Imgs  NC4K/GT
  splits/*.txt
```

Standard protocol: train on CAMO-Train (1,000) and COD10K-Train (3,040). Evaluate on CAMO-Test (250), COD10K-Test (2,026), CHAMELEON (76), and NC4K (4,121). CHAMELEON and NC4K must not be used for training or validation.

Each manifest line is `relative/or/absolute/image.jpg relative/or/absolute/mask.png`.

Only two manifests are authoritative for training:

- `train_camo_cod10k.txt`: the training subset drawn from the 4,040-pair CAMO-Train + COD10K-Train pool;
- `valid_camo_cod10k.txt`: a disjoint held-out subset drawn from the same pool.

Do not create training manifests from CHAMELEON or NC4K. Before training, run:

```bash
python scripts/check_splits.py \
  datasets/splits/train_camo_cod10k.txt \
  datasets/splits/valid_camo_cod10k.txt
```
