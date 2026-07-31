"""Canonical Split Convolutional Attention decoder block."""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="LWCamo")
class SplitConvolutionalAttention(tf.keras.layers.Layer):
    """Split-convolve, multiplicatively gate, and upsample a feature map."""

    def __init__(self, reduction_ratio=4, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = int(reduction_ratio)

    def build(self, input_shape):
        channels = int(input_shape[-1])
        if channels < 8 or channels % 8 != 0:
            raise ValueError(f"SCA channels must be divisible by 8; received {channels}")
        branch_output = channels // 8
        merged_channels = channels // 2
        attention_hidden = max(channels // self.reduction_ratio, 1)
        self.split_convolutions = [
            tf.keras.layers.Conv2D(
                branch_output, 3, padding="same", activation="relu",
                name=f"split_conv_{index + 1}",
            )
            for index in range(4)
        ]
        self.pool = tf.keras.layers.GlobalAveragePooling2D(name="gap")
        self.reduce = tf.keras.layers.Dense(attention_hidden, activation="relu",
                                            name="attention_reduce")
        self.expand = tf.keras.layers.Dense(merged_channels, activation="sigmoid",
                                            name="attention_expand")
        self.reshape = tf.keras.layers.Reshape((1, 1, merged_channels))
        self.upsample = tf.keras.layers.Conv2DTranspose(
            merged_channels, 3, strides=2, padding="same", use_bias=False,
            name="upsample",
        )
        self.normalization = tf.keras.layers.BatchNormalization(name="batch_norm")
        self.activation = tf.keras.layers.ReLU(name="relu")
        super().build(input_shape)

    def call(self, inputs, training=None):
        groups = tf.split(inputs, 4, axis=-1)
        transformed = [layer(group) for layer, group in zip(self.split_convolutions, groups)]
        merged = tf.concat(transformed, axis=-1)
        attention = self.reshape(self.expand(self.reduce(self.pool(merged))))
        attended = merged * attention
        output = self.upsample(attended)
        output = self.normalization(output, training=training)
        return self.activation(output)

    def get_config(self):
        return {**super().get_config(), "reduction_ratio": self.reduction_ratio}
