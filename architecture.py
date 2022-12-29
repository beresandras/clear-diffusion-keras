import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def get_augmenter(uncropped_image_size, image_size):
    return keras.Sequential(
        [
            keras.Input(shape=(uncropped_image_size, uncropped_image_size, 3)),
            layers.Normalization(),
            layers.RandomFlip(mode="horizontal"),
            layers.RandomCrop(height=image_size, width=image_size),
        ],
        name="augmenter",
    )


def get_network(
    image_size,
    noise_embedding_max_frequency,
    noise_embedding_dims,
    image_embedding_dims,
    block_depth,
    widths,
    attentions,
):
    def EmbeddingLayer(embedding_max_frequency, embedding_dims):
        def sinusoidal_embedding(x):
            embedding_min_frequency = 1.0
            frequencies = tf.exp(
                tf.linspace(
                    tf.math.log(embedding_min_frequency),
                    tf.math.log(embedding_max_frequency),
                    embedding_dims // 2,
                )
            )
            angular_speeds = 2.0 * math.pi * frequencies
            embeddings = tf.concat(
                [
                    tf.sin(angular_speeds * x),
                    tf.cos(angular_speeds * x),
                ],
                axis=3,
            )
            return embeddings

        def forward(x):
            x = layers.Lambda(sinusoidal_embedding)(x)
            return x

        return forward

    def ResidualBlock(width, attention):
        def forward(x):
            input_width = x.shape[3]
            if input_width == width:
                residual = x
            else:
                residual = layers.Conv2D(width, kernel_size=1)(x)
            x = layers.BatchNormalization(center=False, scale=False)(x)
            x = layers.Conv2D(
                width, kernel_size=3, padding="same", activation=keras.activations.swish
            )(x)
            if attention:
                x = layers.MultiHeadAttention(
                    num_heads=1, key_dim=width, attention_axes=(1, 2)
                )(x, x)
            else:
                x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
            x = layers.Add()([residual, x])
            return x

        return forward

    def DownBlock(block_depth, width, attention):
        def forward(x):
            x, skips = x
            for _ in range(block_depth):
                x = ResidualBlock(width, attention)(x)
                skips.append(x)
            x = layers.AveragePooling2D(pool_size=2)(x)
            return x

        return forward

    def UpBlock(block_depth, width, attention):
        def forward(x):
            x, skips = x
            x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
            for _ in range(block_depth):
                x = layers.Concatenate()([x, skips.pop()])
                x = ResidualBlock(width, attention)(x)
            return x

        return forward

    images = keras.Input(shape=(image_size, image_size, 3))
    noise_powers = keras.Input(shape=(1, 1, 1))

    x = layers.Conv2D(image_embedding_dims, kernel_size=1)(images)
    skips = [x]

    n = EmbeddingLayer(noise_embedding_max_frequency, noise_embedding_dims)(
        noise_powers
    )
    n = layers.UpSampling2D(size=image_size, interpolation="nearest")(n)
    x = layers.Concatenate()([x, n])

    for width, attention in zip(widths[:-1], attentions[:-1]):
        x = DownBlock(block_depth, width, attention)([x, skips])

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1], attentions[-1])(x)

    for width, attention in zip(widths[-2::-1], attentions[-2::-1]):
        x = UpBlock(block_depth, width, attention)([x, skips])

    x = layers.Concatenate()([x, skips.pop()])
    x = layers.Conv2D(3, kernel_size=1, kernel_initializer="zeros")(x)

    return keras.Model([images, noise_powers], x, name="residual_unet")
