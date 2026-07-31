# Hardware and timing record

Hardware, training, and timing record

## Confirmed environment

- Platform: Google Colab Pro
- GPU model: NVIDIA A100
- GPU memory: 40 GB VRAM
- Numerical precision: FP32
- Input image and mask size: 512 x 512
- Deep-learning framework: TensorFlow
- TensorFlow version: to be retrieved from the experiment runtime
- Python version: to be retrieved from the experiment runtime
- NVIDIA driver version: to be retrieved from the experiment runtime
- CUDA version: to be retrieved from the experiment runtime
- cuDNN version: to be retrieved from the experiment runtime

## Confirmed training configuration

- Model variants: LWCamo-S (MiT-B0), LWCamo-M (MiT-B2), and LWCamo-L (MiT-B5)
- Encoder initialization: ImageNet-pretrained `nvidia/mit-b0`, `nvidia/mit-b2`, and `nvidia/mit-b5`
- Training datasets: CAMO-Train and COD10K-Train only
- Test-only datasets: CAMO-Test, COD10K-Test, CHAMELEON, and NC4K
- Optimizer: Adam
- Initial learning rate: 6 x 10^-5
- Training batch size: 8
- Maximum epochs: 400
- Early stopping: enabled
- Random seed: 42
- External post-processing: none

## Inference timing protocol

- Inference batch size: 1
- Input size: 512 x 512
- Numerical precision: FP32
- Inference mode: enabled
- Warm-up iterations: at least 50
- Timed iterations: at least 300
- Device synchronization: enabled through output tensor materialization
- Disk loading, image decoding, and prediction saving excluded: yes
- Mean latency (ms): to be measured for each model variant
- Latency standard deviation (ms): to be measured for each model variant
- FPS: calculate as `1000 / mean latency in ms`
- Peak GPU inference memory: to be measured for each model variant

## Complexity-reporting convention

- Report complete-model and decoder-only complexity separately.
- Report MACs and FLOPs separately.
- Counting convention: 1 MAC = 2 FLOPs.
- Input tensor for profiling: `1 x 3 x 512 x 512`.
- FLOP-counting tool and version: to be recorded when profiling is performed.
- Batch normalization, activation, attention multiplication, transposed convolution, and resizing treatment: document whether each operator is included by the selected profiler.
- LWCamo-S MACs/FLOPs: to be calculated from the complete executable model.
- LWCamo-M MACs/FLOPs: to be calculated from the complete executable model.
- LWCamo-L MACs/FLOPs: to be calculated from the complete executable model.

## FP32 model-weight storage

Weight storage is calculated as four bytes per parameter and must not be reported as peak runtime memory.

| Variant | Parameters | Approximate FP32 weight storage |
|---|---:|---:|
| LWCamo-S | 3.6 M | 14.4 MB |
| LWCamo-M | 25.5 M | 102.0 MB |
| LWCamo-L | 82.7 M | 330.8 MB |
