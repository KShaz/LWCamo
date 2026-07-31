"""Measure synchronized batch-1 inference latency under a fixed protocol."""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lwcamo.losses import IDBLoss
from lwcamo.models import SplitConvolutionalAttention


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=300)
    args = parser.parse_args()
    model = tf.keras.models.load_model(
        args.model,
        custom_objects={"IDBLoss": IDBLoss,
                        "SplitConvolutionalAttention": SplitConvolutionalAttention},
        compile=False,
    )
    sample = tf.zeros((1, 3, args.image_size, args.image_size), tf.float32)
    call = tf.function(lambda value: model({"pixel_values": value}, training=False))
    for _ in range(args.warmup):
        float(tf.reduce_sum(call(sample)).numpy())
    latencies = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        output = call(sample)
        float(tf.reduce_sum(output).numpy())
        latencies.append((time.perf_counter() - start) * 1000.0)
    mean = statistics.mean(latencies)
    result = {
        "batch_size": 1,
        "image_size": args.image_size,
        "warmup_iterations": args.warmup,
        "timed_iterations": args.iterations,
        "mean_latency_ms": mean,
        "std_latency_ms": statistics.pstdev(latencies),
        "fps": 1000.0 / mean,
        "tensorflow": tf.__version__,
        "gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
    }
    destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
