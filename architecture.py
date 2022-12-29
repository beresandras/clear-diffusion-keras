import math
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers


def get_augmenter(uncropped_image_size, image_size):
    return keras.Sequential(
        [
            keras.Input(shape=(uncropped_image_size, uncropped_image_size, 3)),
            # layers.Normalization(),
            layers.RandomFlip(mode="horizontal"),
            # layers.RandomCrop(height=image_size, width=image_size),
        ],
        name="augmenter",
    )


def get_network(
    image_size,
    noise_embedding_max_frequency,
    noise_embedding_dims,
    image_embedding_dims,
    widths,
    block_depth,
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

    def ResidualBlock(width):
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
    noise_powers = keras.Input(shape=(1, 1, 1))

    x = layers.Conv2D(image_embedding_dims, kernel_size=1)(images)
    skips = [x]

    n = EmbeddingLayer(noise_embedding_max_frequency, noise_embedding_dims)(
        noise_powers
    )
    n = layers.UpSampling2D(size=image_size, interpolation="nearest")(n)
    x = layers.Concatenate()([x, n])

    for width in widths[:-1]:
        x = DownBlock(block_depth, width)([x, skips])

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x)

    for width in reversed(widths[:-1]):
        x = UpBlock(block_depth, width)([x, skips])

    x = layers.Concatenate()([x, skips.pop()])
    x = layers.Conv2D(3, kernel_size=1, kernel_initializer="zeros")(x)

    return keras.Model([images, noise_powers], x, name="residual_unet")

# following based on: https://keras.io/examples/generative/ddpm/#network-architecture

def kernel_init(scale):
    scale = max(scale, 1e-10)
    return keras.initializers.VarianceScaling(
        scale, mode="fan_avg", distribution="uniform"
    )


class AttentionBlock(layers.Layer):
    def __init__(self, units, groups, **kwargs):
        self.units = units
        self.groups = groups
        super().__init__(**kwargs)

        self.norm = tfa.layers.GroupNormalization(groups=groups)
        self.query = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.key = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.value = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.proj = layers.Dense(units, kernel_initializer=kernel_init(0.0))

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "groups": self.groups})
        return config

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        scale = tf.cast(self.units, tf.float32) ** (-0.5)

        inputs = self.norm(inputs)
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        attn_score = tf.einsum("bhwc, bHWc->bhwHW", q, k) * scale
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height * width])

        attn_score = tf.nn.softmax(attn_score, -1)
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height, width])

        proj = tf.einsum("bhwHW,bHWc->bhwc", attn_score, v)
        proj = self.proj(proj)
        return inputs + proj


class TimeEmbedding(layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.half_dim = dim // 2
        self.emb = math.log(10000) / (self.half_dim - 1)
        self.emb = tf.exp(tf.range(self.half_dim, dtype=tf.float32) * -self.emb)

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim})
        return config

    def call(self, inputs):
        inputs = tf.round(1000.0 * inputs[:, 0, 0, 0]) - 1
        emb = inputs[:, None] * self.emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb


def get_network_big(
    image_size,
    noise_embedding_max_frequency,
    noise_embedding_dims,
    image_embedding_dims,
    widths,
    block_depth,
    has_attention=[False, False, True, True],
    norm_groups=8,
    interpolation="nearest",
    activation_fn=keras.activations.swish,
):
    def ResidualBlock(width, groups, activation_fn):
        def apply(inputs):
            x, t = inputs
            input_width = x.shape[3]

            if input_width == width:
                residual = x
            else:
                residual = layers.Conv2D(
                    width, kernel_size=1, kernel_initializer=kernel_init(1.0)
                )(x)

            temb = activation_fn(t)
            temb = layers.Dense(width, kernel_initializer=kernel_init(1.0))(temb)[
                :, None, None, :
            ]

            x = tfa.layers.GroupNormalization(groups=groups)(x)
            x = activation_fn(x)
            x = layers.Conv2D(
                width,
                kernel_size=3,
                padding="same",
                kernel_initializer=kernel_init(1.0),
            )(x)

            x = layers.Add()([x, temb])
            x = tfa.layers.GroupNormalization(groups=groups)(x)
            x = activation_fn(x)

            x = layers.Conv2D(
                width,
                kernel_size=3,
                padding="same",
                kernel_initializer=kernel_init(0.0),
            )(x)
            x = layers.Add()([x, residual])
            return x

        return apply

    def DownSample(width):
        def apply(x):
            x = layers.Conv2D(
                width,
                kernel_size=3,
                strides=2,
                padding="same",
                kernel_initializer=kernel_init(1.0),
            )(x)
            return x

        return apply

    def UpSample(width, interpolation):
        def apply(x):
            x = layers.UpSampling2D(size=2, interpolation=interpolation)(x)
            x = layers.Conv2D(
                width,
                kernel_size=3,
                padding="same",
                kernel_initializer=kernel_init(1.0),
            )(x)
            return x

        return apply

    def TimeMLP(units, activation_fn):
        def apply(inputs):
            temb = layers.Dense(
                units, activation=activation_fn, kernel_initializer=kernel_init(1.0)
            )(inputs)
            temb = layers.Dense(units, kernel_initializer=kernel_init(1.0))(temb)
            return temb

        return apply

    image_input = layers.Input(shape=(image_size, image_size, 3), name="image_input")
    time_input = keras.Input(shape=(1, 1, 1), name="time_input")

    x = layers.Conv2D(
        image_embedding_dims,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer=kernel_init(1.0),
    )(image_input)

    temb = TimeEmbedding(dim=noise_embedding_dims)(time_input)
    temb = TimeMLP(units=noise_embedding_dims, activation_fn=activation_fn)(temb)

    skips = [x]

    # DownBlock
    for i in range(len(widths)):
        for _ in range(block_depth):
            x = ResidualBlock(
                widths[i], groups=norm_groups, activation_fn=activation_fn
            )([x, temb])
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)
            skips.append(x)

        if widths[i] != widths[-1]:
            x = DownSample(widths[i])(x)
            skips.append(x)

    # MiddleBlock
    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn)(
        [x, temb]
    )
    x = AttentionBlock(widths[-1], groups=norm_groups)(x)
    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn)(
        [x, temb]
    )

    # UpBlock
    for i in reversed(range(len(widths))):
        for _ in range(block_depth + 1):
            x = layers.Concatenate(axis=-1)([x, skips.pop()])
            x = ResidualBlock(
                widths[i], groups=norm_groups, activation_fn=activation_fn
            )([x, temb])
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)

        if i != 0:
            x = UpSample(widths[i], interpolation=interpolation)(x)

    # End block
    x = tfa.layers.GroupNormalization(groups=norm_groups)(x)
    x = activation_fn(x)
    x = layers.Conv2D(3, (3, 3), padding="same", kernel_initializer=kernel_init(0.0))(x)
    return keras.Model([image_input, time_input], x, name="unet")
