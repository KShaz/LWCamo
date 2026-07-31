"""Integrated Dice, BCE, and IoU loss."""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="LWCamo")
class IDBLoss(tf.keras.losses.Loss):
    """IDB = 0.5 BCE + 2.0 Dice loss + 1.0 IoU loss."""

    def __init__(self, smooth=1e-6, name="idb_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.smooth = float(smooth)
        self.binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def call(self, y_true, y_pred):
        y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
        axes = (1, 2, 3)
        intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
        true_area = tf.reduce_sum(y_true, axis=axes)
        pred_area = tf.reduce_sum(y_pred, axis=axes)
        dice = (2.0 * intersection + self.smooth) / (
            true_area + pred_area + self.smooth
        )
        iou = (intersection + self.smooth) / (
            true_area + pred_area - intersection + self.smooth
        )
        bce_loss = self.binary_cross_entropy(y_true, y_pred)
        return 0.5 * bce_loss + 2.0 * tf.reduce_mean(1.0 - dice) + tf.reduce_mean(1.0 - iou)

    def get_config(self):
        return {**super().get_config(), "smooth": self.smooth}
