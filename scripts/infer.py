"""Generate unthresholded grayscale probability maps."""

import argparse
from pathlib import Path
import sys
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lwcamo.losses import IDBLoss
from lwcamo.models import SplitConvolutionalAttention


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-size", type=int, default=512)
    args = parser.parse_args()
    model = tf.keras.models.load_model(
        args.model,
        custom_objects={"IDBLoss": IDBLoss,
                        "SplitConvolutionalAttention": SplitConvolutionalAttention},
        compile=False,
    )
    source, output = Path(args.images), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for image_path in sorted(path for path in source.iterdir() if path.is_file()):
        image = tf.io.decode_image(tf.io.read_file(str(image_path)), channels=3,
                                   expand_animations=False)
        image = tf.cast(tf.image.resize(image, (args.image_size, args.image_size)), tf.float32) / 255.0
        model_input = tf.transpose(image, (2, 0, 1))[None, ...]
        probability = model.predict({"pixel_values": model_input}, verbose=0)[0]
        tf.keras.utils.save_img(output / f"{image_path.stem}.png", probability, scale=False)


if __name__ == "__main__":
    main()
