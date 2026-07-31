# Results

Generated outputs belong here and are excluded from source control except for small JSON/CSV summaries.

Recommended structure:

```text
results/<experiment>/best.keras
results/<experiment>/history.csv
results/<experiment>/resolved_config.json
results/predictions/<DATASET>/*.png
results/metrics/<dataset>.json
results/benchmarks/<variant>.json
```

Do not commit checkpoints, raw prediction maps, or large training artifacts to the source branch. Publish checkpoints and prediction archives as versioned release assets with SHA-256 checksums. Small JSON/CSV summaries may be committed after their provenance is recorded.
