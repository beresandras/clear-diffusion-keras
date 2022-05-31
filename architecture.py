import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


def get_augmenter(image_size):
    return keras.Sequential(
        [
            layers.InputLayer(input_shape=(image_size, image_size, 3)),
            preprocessing.RandomFlip(mode="horizontal"),
        ],
        name="augmenter",
    )


def get_network(image_size, num_resolutions, block_depth, width):
    def EmbeddingLayer(width, min_frequency=1.0, max_frequency=1000.0):
        def sinusoidal_embedding(x):
            frequencies = tf.exp(
                tf.linspace(
                    tf.math.log(min_frequency),
                    tf.math.log(max_frequency),
                    width // 2,
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

    def ResidualBlock(width):
        def forward(x):
            input_width = x.shape[3]
            if input_width == width:
                residual = x
            else:
                residual = layers.Conv2D(width, kernel_size=1)(x)
            x = layers.Conv2D(
                width, kernel_size=3, padding="same", activation=keras.activations.swish
            )(x)
            x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
            x = layers.Add()([residual, x])
            return x

        return forward

    def DownBlock(block_depth, width):
        def forward(x):
            x, skips = x
            for _ in range(block_depth):
                x = ResidualBlock(width)(x)
                skips.append(x)
            x = layers.AveragePooling2D(pool_size=2)(x)
            return x

        return forward

    def UpBlock(block_depth, width):
        def forward(x):
            x, skips = x
            x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
            for _ in range(block_depth):
                x = layers.Concatenate()([x, skips.pop()])
                x = ResidualBlock(width)(x)
            return x

        return forward

    images = keras.Input(shape=(image_size, image_size, 3))
    noise_rates = keras.Input(shape=(1, 1, 1))

    x = layers.Conv2D(width, kernel_size=1)(images)
    skips = [x]

    n = EmbeddingLayer(width)(noise_rates)
    n = layers.UpSampling2D(size=image_size, interpolation="nearest")(n)
    x = layers.Concatenate()([x, n])

    for i in range(num_resolutions):
        x = DownBlock(block_depth, (i + 1) * width)([x, skips])

    for _ in range(block_depth):
        x = ResidualBlock((num_resolutions + 1) * width)(x)

    for i in reversed(range(num_resolutions)):
        x = UpBlock(block_depth, (i + 1) * width)([x, skips])

    x = layers.Concatenate()([x, skips.pop()])  # skips is empty after that
    x = layers.Conv2D(3, kernel_size=1)(x)

    return keras.Model([images, noise_rates], x, name="residual_unet")