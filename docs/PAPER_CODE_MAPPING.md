# Paper-to-code mapping

| Manuscript concept | Implementation |
|---|---|
| MiT-B0/B2/B5 encoder | `lwcamo/models/lwcamo.py` |
| B2/B5 512-to-256 projection | `build_lwcamo()` / `decoder_projection` |
| Five-stage SCA decoder | `build_lwcamo()` loop over stages 1-5 |
| Four-way channel split | `lwcamo/models/sca.py` |
| Multiplicative channel attention | `attended = merged * attention` |
| Transposed convolution + BN + ReLU | `SplitConvolutionalAttention.call()` |
| Weighted IDB objective | `lwcamo/losses/idb.py` |
| Paired preprocessing | `lwcamo/data/pairs.py` |
| COD evaluation metrics | `lwcamo/evaluation/cod_metrics.py` |
| Training protocol | `scripts/train.py` + `configs/*.json` |
| CAMO+COD10K-only training manifests | `datasets/splits/train_camo_cod10k.txt` and `valid_camo_cod10k.txt` |
| CHAMELEON/NC4K test-only evaluation | `experiments/evaluate_all.py` |
| Synchronized latency protocol | `scripts/benchmark.py` |

Legacy additive/residual attention implementations are not part of this repository. The authoritative SCA equation is `F = M * A`.
