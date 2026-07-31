"""Leakage-resistant paired image/mask input pipeline."""

from pathlib import Path
import tensorflow as tf


def read_pairs(manifest):
    manifest = Path(manifest).resolve()
    pairs = []
    for line_number, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.replace(",", " ").split()
        if len(fields) != 2:
            raise ValueError(f"{manifest}:{line_number}: expected image and mask paths")
        image, mask = (Path(field) for field in fields)
        image = image if image.is_absolute() else manifest.parent / image
        mask = mask if mask.is_absolute() else manifest.parent / mask
        if not image.is_file() or not mask.is_file():
            raise FileNotFoundError(f"Missing pair: {image} | {mask}")
        pairs.append((str(image), str(mask)))
    if not pairs:
        raise ValueError(f"No pairs found in {manifest}")
    return pairs


def _decode(path, channels):
    value = tf.io.decode_image(tf.io.read_file(path), channels=channels,
                               expand_animations=False)
    value.set_shape((None, None, channels))
    return value


def preprocess_pair(image_path, mask_path, image_size=512, augment=False):
    image = tf.image.resize(_decode(image_path, 3), (image_size, image_size), "bilinear")
    mask = tf.image.resize(_decode(mask_path, 1), (image_size, image_size), "nearest")
    image = tf.cast(image, tf.float32) / 255.0
    mask = tf.cast(tf.cast(mask, tf.float32) / 255.0 >= 0.5, tf.float32)
    if augment:
        seed = tf.random.uniform((2,), maxval=2**31 - 1, dtype=tf.int32)
        image = tf.image.stateless_random_flip_left_right(image, seed)
        mask = tf.image.stateless_random_flip_left_right(mask, seed)
    return {"pixel_values": tf.transpose(image, (2, 0, 1))}, mask


def make_dataset(pairs, batch_size=8, image_size=512, training=False, seed=42):
    images, masks = zip(*pairs)
    dataset = tf.data.Dataset.from_tensor_slices((list(images), list(masks)))
    if training:
        dataset = dataset.shuffle(len(pairs), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.map(
        lambda image, mask: preprocess_pair(image, mask, image_size, training),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
