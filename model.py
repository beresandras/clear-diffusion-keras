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
        prediction_type,
        loss_type,
        batch_size,
        ema,
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
        self.prediction_type = prediction_type
        self.loss_type = loss_type
        self.batch_size = batch_size
        self.ema = ema
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

    def get_components(
        self, noisy_images, predictions, signal_rates, noise_rates, prediction_type=None
    ):
        if prediction_type is None:
            prediction_type = self.prediction_type

        if prediction_type == "velocity":
            pred_velocities = predictions
            pred_images = signal_rates * noisy_images - noise_rates * pred_velocities
            pred_noises = noise_rates * noisy_images + signal_rates * pred_velocities
        elif prediction_type == "signal":
            pred_images = predictions
            pred_noises = (noisy_images - signal_rates * pred_images) / noise_rates
            pred_velocities = (signal_rates * noisy_images - pred_images) / noise_rates
        elif prediction_type == "noise":
            pred_noises = predictions
            pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
            pred_velocities = (pred_noises - noise_rates * noisy_images) / signal_rates
        else:
            raise NotImplementedError
        return pred_velocities, pred_images, pred_noises

    def diffusion_schedule(self, diffusion_times):
        start_snr = tf.exp(self.start_log_snr)
        end_snr = tf.exp(self.end_log_snr)

        start_noise_power = 1.0 / (1.0 + start_snr)
        end_noise_power = 1.0 / (1.0 + end_snr)

        if self.schedule_type == "linear":
            noise_powers = start_noise_power + diffusion_times * (
                end_noise_power - start_noise_power
            )

        elif self.schedule_type == "cosine":
            start_angle = tf.asin(start_noise_power ** 0.5)
            end_angle = tf.asin(end_noise_power ** 0.5)
            diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

            noise_powers = tf.sin(diffusion_angles) ** 2

        elif self.schedule_type == "log-snr-linear":
            noise_powers = start_snr ** diffusion_times / (
                start_snr * end_snr ** diffusion_times + start_snr ** diffusion_times
            )

        elif self.schedule_type == "log-noise-linear":
            noise_powers = (
                start_noise_power
                * (end_noise_power / start_noise_power) ** diffusion_times
            )

        elif self.schedule_type == "log-signal-linear":
            noise_powers = (
                1.0
                - (1.0 - start_noise_power)
                * ((1.0 - end_noise_power) / (1.0 - start_noise_power))
                ** diffusion_times
            )

        elif self.schedule_type == "noise-step-linear":
            noise_powers = start_noise_power * (
                end_noise_power / start_noise_power
            ) ** (diffusion_times ** 2)

        elif self.schedule_type == "signal-step-linear":
            noise_powers = 1.0 - (1.0 - start_noise_power) * (
                (1.0 - end_noise_power) / (1.0 - start_noise_power)
            ) ** (diffusion_times ** 2)

        else:
            raise NotImplementedError("Unsupported sampling schedule")

        signal_powers = 1.0 - noise_powers

        signal_rates = signal_powers ** 0.5
        noise_rates = noise_powers ** 0.5
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

            signal_rates, noise_rates = self.diffusion_schedule(diffusion_times)
            predictions = self.ema_network(
                [noisy_images, noise_rates ** 2], training=False
            )
            # pred_images = tf.clip_by_value(pred_images, -1.0, 1.0)
            _, pred_images, pred_noises = self.get_components(
                noisy_images, predictions, signal_rates, noise_rates
            )

            if second_order_alpha is not None:
                # use first estimate to sample alpha steps away
                alpha_signal_rates, alpha_noise_rates = self.diffusion_schedule(
                    diffusion_times - second_order_alpha * step_size
                )
                alpha_noisy_images = (
                    alpha_signal_rates * pred_images + alpha_noise_rates * pred_noises
                )
                alpha_predictions = self.ema_network(
                    [alpha_noisy_images, alpha_noise_rates ** 2], training=False
                )
                # calculate noise estimate from prediction
                _, _, alpha_pred_noises = self.get_components(
                    alpha_noisy_images,
                    alpha_predictions,
                    alpha_signal_rates,
                    alpha_noise_rates,
                )

                # linearly combine the two noise estimates
                pred_noises = (
                    1.0 - 1.0 / (2.0 * second_order_alpha)
                ) * pred_noises + 1.0 / (2.0 * second_order_alpha) * alpha_pred_noises
                _, pred_images, pred_noises = self.get_components(
                    noisy_images,
                    pred_noises,
                    signal_rates,
                    noise_rates,
                    prediction_type="noise",
                )

            next_signal_rates, next_noise_rates = self.diffusion_schedule(
                diffusion_times - step_size
            )
            if stochastic:
                sample_noise_rates = (
                    1.0 - (signal_rates / next_signal_rates) ** 2
                ) ** 0.5 * (next_noise_rates / noise_rates)
                noisy_images = (
                    next_signal_rates * pred_images
                    + (next_noise_rates ** 2 - sample_noise_rates ** 2) ** 0.5
                    * pred_noises
                )

                sample_noises = tf.random.normal(
                    shape=(batch_size, self.image_size, self.image_size, 3)
                )
                if variance_preserving:
                    noisy_images += sample_noise_rates * sample_noises
                else:
                    noisy_images += (
                        sample_noise_rates
                        * (noise_rates / next_noise_rates)
                        * sample_noises
                    )

            else:
                noisy_images = (
                    next_signal_rates * pred_images + next_noise_rates * pred_noises
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
        signal_rates, noise_rates = self.diffusion_schedule(diffusion_times)

        noisy_images = signal_rates * images + noise_rates * noises
        velocities = -noise_rates * images + signal_rates * noises

        with tf.GradientTape() as tape:
            predictions = self.network([noisy_images, noise_rates ** 2], training=True)
            pred_velocities, pred_images, pred_noises = self.get_components(
                noisy_images, predictions, signal_rates, noise_rates
            )

            velocity_loss = self.loss(velocities, pred_velocities)
            image_loss = self.loss(images, pred_images)
            noise_loss = self.loss(noises, pred_noises)

        if self.loss_type == "velocity":
            loss = velocity_loss
        elif self.loss_type == "signal":
            loss = image_loss
        elif self.loss_type == "noise":
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
        signal_rates, noise_rates = self.diffusion_schedule(diffusion_times)

        noisy_images = signal_rates * images + noise_rates * noises
        velocities = -noise_rates * images + signal_rates * noises

        predictions = self.ema_network([noisy_images, noise_rates ** 2], training=False)
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
        num_rows=3,
        num_cols=8,
        diffusion_steps=20,
        stochastic=False,
        variance_preserving=False,
        second_order_alpha=None,
        plot_image_size=128,
    ):
        generated_images = self.generate(
            num_rows * num_cols,
            diffusion_steps,
            stochastic,
            variance_preserving,
            second_order_alpha,
        )

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
            "images/{}_{}_{}_{:.3f}.png".format(
                self.id,
                "final" if epoch is None else epoch + 1,
                "s" if stochastic else "d",
                self.kid.result(),
            ),
            generated_images.numpy(),
        )
