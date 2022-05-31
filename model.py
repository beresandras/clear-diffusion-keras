import math
import matplotlib.pyplot as plt
import tensorflow as tf

from abc import abstractmethod
from tensorflow import keras

from metrics import KID


class DiffusionModel(keras.Model):
    def __init__(
        self,
        id,
        augmenter,
        network,
        batch_size,
        time_margin,
        ema,
        kid_image_size,
        plot_interval,
    ):
        super().__init__()
        self.id = id

        self.augmenter = augmenter
        self.network = network
        self.ema_network = keras.models.clone_model(network)

        self.image_size = network.input_shape[0][1]
        self.batch_size = batch_size
        self.time_margin = time_margin
        self.ema = ema
        self.kid_image_size = kid_image_size
        self.plot_interval = plot_interval

    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss")
        self.kid = KID(
            input_shape=self.network.output_shape[1:], image_size=self.kid_image_size
        )

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker, self.kid]

    @abstractmethod
    def noise_schedule(self, diffusion_times):
        noise_rates = tf.sin(0.5 * diffusion_times * math.pi) ** 2
        return noise_rates

    @abstractmethod
    def denoise(self, noisy_images, noise_rates, training):
        if training:
            network = self.network
        else:
            network = self.ema_network

        pred_noises = network([noisy_images, noise_rates], training=training)
        pred_images = (1.0 - noise_rates) ** -0.5 * (
            noisy_images - noise_rates ** 0.5 * pred_noises
        )

        return pred_images, pred_noises

    def diffusion_process(self, initial_noise, diffusion_steps):
        batch_size = tf.shape(initial_noise)[0]
        diffusion_times = tf.linspace(
            1.0 - self.time_margin, self.time_margin, diffusion_steps + 1
        )
        diffusion_times = tf.reshape(
            diffusion_times, shape=(diffusion_steps + 1, 1, 1, 1, 1)
        )
        diffusion_times = tf.broadcast_to(
            diffusion_times, shape=(diffusion_steps + 1, batch_size, 1, 1, 1)
        )

        noisy_images = initial_noise
        for step in range(diffusion_steps):
            noise_rates = self.noise_schedule(diffusion_times[step])
            next_noise_rates = self.noise_schedule(diffusion_times[step + 1])

            pred_images, pred_noises = self.denoise(
                noisy_images, noise_rates, training=False
            )

            noisy_images = (
                1.0 - next_noise_rates
            ) ** 0.5 * pred_images + next_noise_rates ** 0.5 * pred_noises

        return pred_images

    def generate(self, num_images, diffusion_steps):
        noise_samples = tf.random.normal(
            shape=(num_images, self.image_size, self.image_size, 3)
        )
        return self.diffusion_process(
            initial_noise=noise_samples, diffusion_steps=diffusion_steps
        )

    def train_step(self, images):
        images = self.augmenter(images, training=True)

        noises = tf.random.normal(
            shape=(self.batch_size, self.image_size, self.image_size, 3)
        )
        diffusion_times = tf.random.uniform(
            shape=(self.batch_size, 1, 1, 1),
            minval=self.time_margin,
            maxval=1.0 - self.time_margin,
        )
        noise_rates = self.noise_schedule(diffusion_times)
        noisy_images = (1.0 - noise_rates) ** 0.5 * images + noise_rates ** 0.5 * noises

        with tf.GradientTape() as tape:
            pred_images, pred_noises = self.denoise(
                noisy_images, noise_rates, training=True
            )

            noise_loss = self.loss(noises, pred_noises)
            image_loss = self.loss(images, pred_images)

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        # KID is not measured during the training phase for computational efficiency
        return {m.name: m.result() for m in self.metrics[:-1]}

    def test_step(self, images):
        images = self.augmenter(images, training=False)

        noises = tf.random.normal(
            shape=(self.batch_size, self.image_size, self.image_size, 3)
        )
        diffusion_times = tf.random.uniform(
            shape=(self.batch_size, 1, 1, 1),
            minval=self.time_margin,
            maxval=1.0 - self.time_margin,
        )
        noise_rates = self.noise_schedule(diffusion_times)
        noisy_images = (1.0 - noise_rates) ** 0.5 * images + noise_rates ** 0.5 * noises

        pred_images, pred_noises = self.denoise(
            noisy_images, noise_rates, training=False
        )

        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.loss(images, pred_images)

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        generated_images = self.generate(self.batch_size, diffusion_steps=10)
        self.kid.update_state(images, generated_images)

        return {m.name: m.result() for m in self.metrics}

    def plot_images(self, epoch=-1, logs=None, num_rows=4, num_cols=8):
        if (epoch + 1) % self.plot_interval == 0:
            num_images = num_rows * num_cols

            generated_images = self.generate(num_images, diffusion_steps=10)
            generated_images = 0.5 * (1.0 + generated_images)
            generated_images = tf.clip_by_value(generated_images, 0.0, 1.0)

            plt.figure(figsize=(num_cols * 1.5, num_rows * 1.5))
            for row in range(num_rows):
                for col in range(num_cols):
                    index = row * num_cols + col
                    plt.subplot(num_rows, num_cols, index + 1)
                    plt.imshow(generated_images[index])
                    plt.axis("off")
            plt.tight_layout()
            plt.savefig(
                "images/{}_{}_{:.3f}.png".format(self.id, epoch + 1, self.kid.result())
            )
            plt.close()