"""Reproducibility utilities."""

import random
import numpy as np
import tensorflow as tf


def configure_reproducibility(seed=42, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    if deterministic:
        tf.config.experimental.enable_op_determinism()
