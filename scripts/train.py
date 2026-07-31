"""Train one LWCamo variant from disjoint train/validation manifests."""

import argparse
import json
from pathlib import Path
import sys
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lwcamo.data import make_dataset, read_pairs
from lwcamo.losses import IDBLoss
from lwcamo.models import build_lwcamo
from lwcamo.utils import configure_reproducibility


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--valid-list", required=True)
    parser.add_argument("--output-dir", default="results/experiment")
    return parser.parse_args()


def main():
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    configure_reproducibility(config["seed"], config.get("deterministic", True))
    train_pairs, valid_pairs = read_pairs(args.train_list), read_pairs(args.valid_list)
    overlap = {image for image, _ in train_pairs} & {image for image, _ in valid_pairs}
    if overlap:
        raise ValueError(f"Train/validation leakage: {len(overlap)} shared images")
    train_data = make_dataset(train_pairs, config["batch_size"], config["image_size"], True,
                              config["seed"])
    valid_data = make_dataset(valid_pairs, config["batch_size"], config["image_size"], False,
                              config["seed"])
    model = build_lwcamo(config["variant"], config["image_size"], config["pretrained"])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config["learning_rate"]),
        loss=IDBLoss(),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    with (output / "model_summary.txt").open("w", encoding="utf-8") as summary:
        model.summary(print_fn=lambda line: summary.write(line + "\n"))
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(output / "best.keras"), monitor="val_loss",
                                           mode="min", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10,
                                             min_lr=1e-7),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=30,
                                         restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(str(output / "history.csv")),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    model.fit(train_data, validation_data=valid_data, epochs=config["epochs"], callbacks=callbacks)
    (output / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
