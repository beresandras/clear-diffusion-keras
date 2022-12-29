import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


class KID(keras.metrics.Metric):
    def __init__(self, name="kid", input_shape=None, image_size=None, **kwargs):
        super().__init__(name=name, **kwargs)

        # resolution of images for the KID estimation
        self.image_size = image_size

        # KID is estimated per batch and is averaged across batches
        self.kid_tracker = keras.metrics.Mean()

        # a pretrained InceptionV3 is used without its classification layer
        self.encoder = keras.Sequential(
            [
                keras.Input(shape=input_shape),
                layers.Lambda(self.preprocess_input),
                keras.applications.InceptionV3(
                    include_top=False,
                    input_shape=(self.image_size, self.image_size, 3),
                    weights="imagenet",
                ),
                layers.GlobalAveragePooling2D(),
            ]
        )

    def preprocess_input(self, images):
        images = tf.image.resize(
            images,
            size=[self.image_size, self.image_size],
            method="bicubic",
            antialias=True,
        )
        images = tf.clip_by_value(images, 0.0, 1.0)
        images = keras.applications.inception_v3.preprocess_input(images * 255.0)
        return images

    def polynomial_kernel(self, features_1, features_2):
        feature_dimensions = tf.cast(tf.shape(features_1)[1], dtype=tf.float32)
        return (features_1 @ tf.transpose(features_2) / feature_dimensions + 1.0) ** 3.0

    def update_state(self, real_images, generated_images, sample_weight=None):
        real_features = self.encoder(real_images, training=False)
        generated_features = self.encoder(generated_images, training=False)

        # compute polynomial kernels using the two sets of features
        kernel_real = self.polynomial_kernel(real_features, real_features)
        kernel_generated = self.polynomial_kernel(
            generated_features, generated_features
        )
        kernel_cross = self.polynomial_kernel(real_features, generated_features)

        # estimate the squared maximum mean discrepancy using the average kernel values
        batch_size = tf.shape(real_features)[0]
        batch_size_f = tf.cast(batch_size, dtype=tf.float32)
        mean_kernel_real = tf.reduce_sum(kernel_real * (1.0 - tf.eye(batch_size))) / (
            batch_size_f * (batch_size_f - 1.0)
        )
        mean_kernel_generated = tf.reduce_sum(
            kernel_generated * (1.0 - tf.eye(batch_size))
        ) / (batch_size_f * (batch_size_f - 1.0))
        mean_kernel_cross = tf.reduce_mean(kernel_cross)
        kid = mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross

        # update the average KID estimate
        self.kid_tracker.update_state(kid)

        return kid  # this is not necessary but useful for debugging

    def result(self):
        return self.kid_tracker.result()

    def reset_state(self):
        self.kid_tracker.reset_state()
