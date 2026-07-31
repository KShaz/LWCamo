"""Model factory for LWCamo-S, LWCamo-M, and LWCamo-L."""

import tensorflow as tf
from transformers import SegformerConfig, TFSegformerModel

from .sca import SplitConvolutionalAttention


BACKBONES = {
    "small": "nvidia/mit-b0",
    "medium": "nvidia/mit-b2",
    "large": "nvidia/mit-b5",
}


def build_lwcamo(variant="small", image_size=512, pretrained=True):
    if variant not in BACKBONES:
        raise ValueError(f"Unknown variant {variant!r}; choose from {tuple(BACKBONES)}")
    checkpoint = BACKBONES[variant]
    if pretrained:
        encoder = TFSegformerModel.from_pretrained(checkpoint)
    else:
        encoder = TFSegformerModel(SegformerConfig.from_pretrained(checkpoint))

    inputs = tf.keras.Input((3, image_size, image_size), name="pixel_values")
    deepest = encoder(inputs, return_dict=True).last_hidden_state
    deepest = tf.keras.layers.Permute((2, 3, 1), name="channels_last")(deepest)

    # B0 emits 256 channels; B2/B5 emit 512. The paper defines one 256-channel
    # decoder interface, so larger variants use a learned 1x1 projection.
    if int(deepest.shape[-1]) != 256:
        deepest = tf.keras.layers.Conv2D(
            256, 1, padding="same", name="decoder_projection"
        )(deepest)

    decoded = deepest
    for stage in range(1, 6):
        decoded = SplitConvolutionalAttention(name=f"sca_{stage}")(decoded)
    probability = tf.keras.layers.Conv2D(
        1, 1, padding="same", activation="sigmoid", name="probability_map"
    )(decoded)
    if probability.shape[1] != image_size or probability.shape[2] != image_size:
        probability = tf.keras.layers.Resizing(
            image_size, image_size, interpolation="bilinear", name="output_resize"
        )(probability)
    return tf.keras.Model(inputs, probability, name=f"LWCamo-{variant[0].upper()}")
