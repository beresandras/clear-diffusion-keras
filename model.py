import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras

from metrics import KID


class DiffusionModel(keras.Model):
    def __init__(
        self,
        id,
        augmenter,
        network,
        batch_size,
        ema,
        output_type,
        schedule_type,
        start_log_snr,
        end_log_snr,
        kid_image_size,
        kid_diffusion_steps,
    ):
        super().__init__()
        self.id = id

        self.augmenter = augmenter
        self.network = network
        self.ema_network = keras.models.clone_model(network)

        self.image_size = network.input_shape[0][1]
        self.batch_size = batch_size
        self.ema = ema
        self.output_type = output_type
        self.schedule_type = schedule_type
        self.start_log_snr = start_log_snr
        self.end_log_snr = end_log_snr
        self.kid_image_size = kid_image_size
        self.kid_diffusion_steps = kid_diffusion_steps

    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.velocity_loss_tracker = keras.metrics.Mean(name="v_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss")
        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        self.kid = KID(
            input_shape=self.network.output_shape[1:], image_size=self.kid_image_size
        )

    @property
    def metrics(self):
        return [
            self.velocity_loss_tracker,
            self.image_loss_tracker,
            self.noise_loss_tracker,
            self.kid,
        ]

    def denormalize(self, images):
        images = self.augmenter.layers[0].mean + (
            images * self.augmenter.layers[0].variance ** 0.5
        )
        return tf.clip_by_value(images, 0.0, 1.0)

    def get_components(self, noisy_images, predictions, signal_rates, noise_rates):
        if self.output_type == "velocity":
            pred_velocities = predictions
            pred_images = (
                noisy_images * signal_rates ** 0.5
                - pred_velocities * noise_rates ** 0.5
            )
            pred_noises = (
                noisy_images * noise_rates ** 0.5
                + pred_velocities * signal_rates ** 0.5
            )
        elif self.output_type == "signal":
            pred_images = predictions
            pred_noises = noise_rates ** -0.5 * (
                noisy_images - signal_rates ** 0.5 * pred_images
            )
            pred_velocities = noise_rates ** -0.5 * (
                signal_rates ** 0.5 * noisy_images - pred_images
            )
        elif self.output_type == "noise":
            pred_noises = predictions
            pred_images = signal_rates ** -0.5 * (
                noisy_images - noise_rates ** 0.5 * pred_noises
            )
            pred_velocities = signal_rates ** -0.5 * (
                pred_noises - noise_rates ** 0.5 * noisy_images
            )
        else:
            raise NotImplementedError
        return pred_velocities, pred_images, pred_noises

    def noise_schedule(self, diffusion_times):
        start_snr = tf.exp(self.start_log_snr)
        end_snr = tf.exp(self.end_log_snr)

        start_noise_rate = 1.0 / (1.0 + start_snr)
        end_noise_rate = 1.0 / (1.0 + end_snr)

        if self.schedule_type == "linear":
            noise_rates = start_noise_rate + diffusion_times * (
                end_noise_rate - start_noise_rate
            )

        elif self.schedule_type == "cosine":
            start_angle = tf.asin(start_noise_rate ** 0.5)
            end_angle = tf.asin(end_noise_rate ** 0.5)
            diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

            noise_rates = tf.sin(diffusion_angles) ** 2

        elif self.schedule_type == "log-snr-linear":
            noise_rates = start_snr ** diffusion_times / (
                start_snr * end_snr ** diffusion_times + start_snr ** diffusion_times
            )

        elif self.schedule_type == "log-noise-linear":
            noise_rates = (
                start_noise_rate
                * (end_noise_rate / start_noise_rate) ** diffusion_times
            )

        elif self.schedule_type == "log-signal-linear":
            noise_rates = (
                1.0
                - (1.0 - start_noise_rate)
                * ((1.0 - end_noise_rate) / (1.0 - start_noise_rate)) ** diffusion_times
            )

        elif self.schedule_type == "noise-step-linear":
            noise_rates = start_noise_rate * (end_noise_rate / start_noise_rate) ** (
                diffusion_times ** 2
            )

        elif self.schedule_type == "signal-step-linear":
            noise_rates = 1.0 - (1.0 - start_noise_rate) * (
                (1.0 - end_noise_rate) / (1.0 - start_noise_rate)
            ) ** (diffusion_times ** 2)

        else:
            raise NotImplementedError("Unsupported sampling schedule")

        signal_rates = 1.0 - noise_rates
        return signal_rates, noise_rates

    def diffusion_process(
        self,
        initial_noise,
        diffusion_steps,
        stochastic,
        variance_preserving,
        second_order_alpha,
    ):
        batch_size = tf.shape(initial_noise)[0]
        step_size = 1.0 / diffusion_steps

        noisy_images = initial_noise
        for step in range(diffusion_steps):
            diffusion_times = tf.ones((batch_size, 1, 1, 1)) - step * step_size

            signal_rates, noise_rates = self.noise_schedule(diffusion_times)
            predictions = self.ema_network([noisy_images, noise_rates], training=False)
            # pred_images = tf.clip_by_value(pred_images, -1.0, 1.0)
            _, pred_images, pred_noises = self.get_components(
                noisy_images, predictions, signal_rates, noise_rates
            )

            if second_order_alpha is not None:
                # use first estimate to sample alpha steps away
                alpha_signal_rates, alpha_noise_rates = self.noise_schedule(
                    diffusion_times - second_order_alpha * step_size
                )
                alpha_noisy_images = (
                    alpha_signal_rates ** 0.5 * pred_images
                    + alpha_noise_rates ** 0.5 * pred_noises
                )
                alpha_predictions = self.ema_network(
                    [alpha_noisy_images, alpha_noise_rates], training=False
                )

                # linearly combine the two estimates
                predictions = (
                    1.0 - 1.0 / (2.0 * second_order_alpha)
                ) * predictions + 1.0 / (2.0 * second_order_alpha) * alpha_predictions
                _, pred_images, pred_noises = self.get_components(
                    noisy_images, predictions, signal_rates, noise_rates
                )

            next_signal_rates, next_noise_rates = self.noise_schedule(
                diffusion_times - step_size
            )
            if stochastic:
                sample_noise_rates = (1.0 - signal_rates / next_signal_rates) * (
                    next_noise_rates / noise_rates
                )
                noisy_images = (
                    next_signal_rates ** 0.5 * pred_images
                    + (next_noise_rates - sample_noise_rates) ** 0.5 * pred_noises
                )

                sample_noises = tf.random.normal(
                    shape=(batch_size, self.image_size, self.image_size, 3)
                )
                if variance_preserving:
                    noisy_images += sample_noise_rates ** 0.5 * sample_noises
                else:
                    noisy_images += (
                        sample_noise_rates * (noise_rates / next_noise_rates)
                    ) ** 0.5 * sample_noises

            else:
                noisy_images = (
                    next_signal_rates ** 0.5 * pred_images
                    + next_noise_rates ** 0.5 * pred_noises
                )

        return pred_images

    def generate(
        self,
        num_images,
        diffusion_steps,
        stochastic,
        variance_preserving,
        second_order_alpha,
    ):
        initial_noise = tf.random.normal(
            shape=(num_images, self.image_size, self.image_size, 3)
        )
        generated_images = self.diffusion_process(
            initial_noise,
            diffusion_steps,
            stochastic,
            variance_preserving,
            second_order_alpha,
        )
        return self.denormalize(generated_images)

    def train_step(self, images):
        images = self.augmenter(images, training=True)
        noises = tf.random.normal(
            shape=(self.batch_size, self.image_size, self.image_size, 3)
        )

        diffusion_times = tf.random.uniform(
            shape=(self.batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        signal_rates, noise_rates = self.noise_schedule(diffusion_times)

        noisy_images = signal_rates ** 0.5 * images + noise_rates ** 0.5 * noises
        velocities = -(noise_rates ** 0.5) * images + signal_rates ** 0.5 * noises

        with tf.GradientTape() as tape:
            predictions = self.network([noisy_images, noise_rates], training=True)
            pred_velocities, pred_images, pred_noises = self.get_components(
                noisy_images, predictions, signal_rates, noise_rates
            )

            velocity_loss = self.loss(velocities, pred_velocities)
            image_loss = self.loss(images, pred_images)
            noise_loss = self.loss(noises, pred_noises)

        if self.output_type == "velocity":
            loss = velocity_loss
        elif self.output_type == "signal":
            loss = image_loss
        elif self.output_type == "noise":
            loss = noise_loss
        else:
            raise NotImplementedError

        gradients = tape.gradient(loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.velocity_loss_tracker.update_state(velocity_loss)
        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)

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
            shape=(self.batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        signal_rates, noise_rates = self.noise_schedule(diffusion_times)

        noisy_images = signal_rates ** 0.5 * images + noise_rates ** 0.5 * noises
        velocities = -(noise_rates ** 0.5) * images + signal_rates ** 0.5 * noises

        predictions = self.ema_network([noisy_images, noise_rates], training=False)
        pred_velocities, pred_images, pred_noises = self.get_components(
            noisy_images, predictions, signal_rates, noise_rates
        )

        velocity_loss = self.loss(velocities, pred_velocities)
        image_loss = self.loss(images, pred_images)
        noise_loss = self.loss(noises, pred_noises)

        self.velocity_loss_tracker.update_state(velocity_loss)
        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        images = self.denormalize(images)
        generated_images = self.generate(
            self.batch_size,
            diffusion_steps=self.kid_diffusion_steps,
            stochastic=False,
            variance_preserving=False,
            second_order_alpha=None,
        )
        self.kid.update_state(images, generated_images)

        return {m.name: m.result() for m in self.metrics}

    def plot_images(
        self,
        epoch=None,
        logs=None,
        num_rows=4,
        num_cols=8,
        diffusion_steps=20,
        stochastic=False,
        variance_preserving=False,
        second_order_alpha=None,
    ):
        generated_images = self.generate(
            num_rows * num_cols,
            diffusion_steps,
            stochastic,
            variance_preserving,
            second_order_alpha,
        )

        plot_image_size = 2 * self.image_size
        generated_images = tf.image.resize(
            generated_images, (plot_image_size, plot_image_size), method="nearest"
        )
        generated_images = tf.reshape(
            generated_images,
            (num_rows, num_cols, plot_image_size, plot_image_size, 3),
        )
        generated_images = tf.transpose(generated_images, (0, 2, 1, 3, 4))
        generated_images = tf.reshape(
            generated_images,
            (num_rows * plot_image_size, num_cols * plot_image_size, 3),
        )
        plt.imsave(
            "images/{}_{}_{:.3f}.png".format(
                self.id, "final" if epoch is None else epoch + 1, self.kid.result()
            ),
            generated_images.numpy(),
        )
