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
        is_jupyter=False,
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
        self.is_jupyter = is_jupyter

        # only required for multistep sampling
        self.multistep_coefficients = [
            tf.constant([1], shape=(1, 1, 1, 1, 1), dtype=tf.float32),
            tf.constant([-1, 3], shape=(2, 1, 1, 1, 1), dtype=tf.float32) / 2,
            tf.constant([5, -16, 23], shape=(3, 1, 1, 1, 1), dtype=tf.float32) / 12,
            tf.constant([-9, 37, -59, 55], shape=(4, 1, 1, 1, 1), dtype=tf.float32)
            / 24,
            tf.constant(
                [251, -1274, 2616, -2774, 1901], shape=(5, 1, 1, 1, 1), dtype=tf.float32
            )
            / 720,
        ]

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
        # convert the pixel values back to 0-1 range
        images = self.augmenter.layers[0].mean + (
            images * self.augmenter.layers[0].variance ** 0.5
        )
        return tf.clip_by_value(images, 0.0, 1.0)

    def get_components(
        self, noisy_images, predictions, signal_rates, noise_rates, prediction_type=None
    ):
        if prediction_type is None:
            prediction_type = self.prediction_type

        # calculate the other signal components using the network prediction
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
            # variance or power of noise component increases linearly
            noise_powers = start_noise_power + diffusion_times * (
                end_noise_power - start_noise_power
            )

        elif self.schedule_type == "cosine":
            # noise rate increases sinusoidally
            # signal rate decreases as a cosine function
            # simplified from the "cosine schedule" of Improved DDPM https://arxiv.org/abs/2102.09672
            start_angle = tf.asin(start_noise_power**0.5)
            end_angle = tf.asin(end_noise_power**0.5)
            diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

            noise_powers = tf.sin(diffusion_angles) ** 2

        elif self.schedule_type == "log-snr-linear":
            # the log signal-to-noise ratio decreases linearly
            # proposed in VDM https://arxiv.org/abs/2107.00630
            noise_powers = start_snr**diffusion_times / (
                start_snr * end_snr**diffusion_times + start_snr**diffusion_times
            )

        else:
            raise NotImplementedError("Unsupported sampling schedule.")

        # the signal and noise power will always sum to one
        signal_powers = 1.0 - noise_powers

        # the rates are the square roots of the powers
        # variance**0.5 -> standard deviation
        signal_rates = signal_powers**0.5
        noise_rates = noise_powers**0.5
        return signal_rates, noise_rates

    def generate(
        self,
        num_images,
        diffusion_steps,
        stochasticity,
        variance_preserving,
        num_multisteps,
        second_order_alpha,
    ):
        assert num_multisteps <= 5, "Maximum 5 multisteps are supported."

        # noise -> images -> denormalized images
        initial_noise = tf.random.normal(
            shape=(num_images, self.image_size, self.image_size, 3)
        )
        generated_images = self.diffusion_process(
            initial_noise,
            diffusion_steps,
            stochasticity,
            variance_preserving,
            num_multisteps,
            second_order_alpha,
        )
        return self.denormalize(generated_images)

    def diffusion_process(
        self,
        initial_noise,
        diffusion_steps,
        stochasticity,
        variance_preserving,
        num_multisteps,
        second_order_alpha,
    ):
        batch_size = tf.shape(initial_noise)[0]
        step_size = 1.0 / diffusion_steps

        # at the first sampling step, the "noisy image" is pure noise
        noisy_images = initial_noise
        prev_pred_noises = []  # only required for multistep sampling
        for step in range(diffusion_steps):
            diffusion_times = tf.ones((batch_size, 1, 1, 1)) - step * step_size

            signal_rates, noise_rates = self.diffusion_schedule(diffusion_times)
            # predict one component of the noisy images with the network
            # exponential moving average weights are used for inference
            predictions = self.ema_network(
                [noisy_images, noise_rates**2], training=False
            )
            # calculate the other components using it
            _, pred_images, pred_noises = self.get_components(
                noisy_images, predictions, signal_rates, noise_rates
            )

            # optional multistep sampling
            if num_multisteps > 1:
                prev_pred_noises.append(pred_noises)
                pred_images, pred_noises = self.multistep_correction(
                    noisy_images,
                    signal_rates,
                    noise_rates,
                    prev_pred_noises,
                    num_multisteps,
                )

            # optional second order sampling
            if second_order_alpha is not None:
                pred_images, pred_noises = self.second_order_correction(
                    diffusion_times,
                    step_size,
                    noisy_images,
                    signal_rates,
                    noise_rates,
                    pred_images,
                    pred_noises,
                    second_order_alpha,
                )

            next_signal_rates, next_noise_rates = self.diffusion_schedule(
                diffusion_times - step_size
            )
            if stochasticity > 0.0:
                # optional stochastic sampling
                noisy_images = self.get_stochastic_noisy_image(
                    signal_rates,
                    noise_rates,
                    next_signal_rates,
                    next_noise_rates,
                    pred_images,
                    pred_noises,
                    stochasticity,
                    variance_preserving,
                )
            else:
                # remix the predicted components using the next signal and noise rates
                noisy_images = (
                    next_signal_rates * pred_images + next_noise_rates * pred_noises
                )
            # this new noisy image will be used in the next step

        return pred_images

    def get_stochastic_noisy_image(
        self,
        signal_rates,
        noise_rates,
        next_signal_rates,
        next_noise_rates,
        pred_images,
        pred_noises,
        stochasticity,
        variance_preserving,
    ):
        sample_noise_rates = (
            stochasticity
            * (1.0 - (signal_rates / next_signal_rates) ** 2) ** 0.5
            * (next_noise_rates / noise_rates)
        )

        # reduce the power of the predicted noise component
        noisy_images = (
            next_signal_rates * pred_images
            + (next_noise_rates**2 - sample_noise_rates**2) ** 0.5 * pred_noises
        )

        # add some amount of random noise instead
        sample_noises = tf.random.normal(shape=tf.shape(pred_noises))
        if variance_preserving:  # same amount
            noisy_images += sample_noise_rates * sample_noises
        else:  # larger amount
            noisy_images += (
                sample_noise_rates * (noise_rates / next_noise_rates) * sample_noises
            )

        return noisy_images

    def multistep_correction(
        self, noisy_images, signal_rates, noise_rates, prev_pred_noises, num_multisteps
    ):
        # Adams-Bashforth multistep method
        # https://en.wikipedia.org/wiki/Linear_multistep_method#Adams%E2%80%93Bashforth_methods
        # based on https://arxiv.org/abs/2202.09778

        # linearly combine previous noise estimates
        # doing this with the image component leads to identical results
        pred_noises = tf.reduce_sum(
            self.multistep_coefficients[len(prev_pred_noises) - 1]
            * tf.stack(prev_pred_noises, axis=0),
            axis=0,
        )
        if len(prev_pred_noises) == num_multisteps:
            # remove oldest noise estimate
            prev_pred_noises.pop(0)

        # recalculate component estimates
        _, pred_images, pred_noises = self.get_components(
            noisy_images,
            pred_noises,
            signal_rates,
            noise_rates,
            prediction_type="noise",
        )
        return pred_images, pred_noises

    def second_order_correction(
        self,
        diffusion_times,
        step_size,
        noisy_images,
        signal_rates,
        noise_rates,
        pred_images,
        pred_noises,
        second_order_alpha,
    ):
        # generic second-order Runge-Kutta method
        # https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Generic_second-order_method
        # based on https://arxiv.org/abs/2206.00364

        # use first estimate to sample alpha steps away
        alpha_signal_rates, alpha_noise_rates = self.diffusion_schedule(
            diffusion_times - second_order_alpha * step_size
        )
        alpha_noisy_images = (
            alpha_signal_rates * pred_images + alpha_noise_rates * pred_noises
        )
        alpha_predictions = self.ema_network(
            [alpha_noisy_images, alpha_noise_rates**2], training=False
        )
        # calculate noise estimate from prediction
        _, _, alpha_pred_noises = self.get_components(
            alpha_noisy_images,
            alpha_predictions,
            alpha_signal_rates,
            alpha_noise_rates,
        )

        # linearly combine the two noise estimates
        pred_noises = (1.0 - 1.0 / (2.0 * second_order_alpha)) * pred_noises + 1.0 / (
            2.0 * second_order_alpha
        ) * alpha_pred_noises

        # recalculate component estimates
        _, pred_images, pred_noises = self.get_components(
            noisy_images,
            pred_noises,
            signal_rates,
            noise_rates,
            prediction_type="noise",
        )
        return pred_images, pred_noises

    def train_step(self, images):
        # normalize images to have standard deviation of 1, like the noises
        images = self.augmenter(images, training=True)
        noises = tf.random.normal(
            shape=(self.batch_size, self.image_size, self.image_size, 3)
        )

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(self.batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        signal_rates, noise_rates = self.diffusion_schedule(diffusion_times)

        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises
        velocities = -noise_rates * images + signal_rates * noises

        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            predictions = self.network([noisy_images, noise_rates**2], training=True)
            pred_velocities, pred_images, pred_noises = self.get_components(
                noisy_images, predictions, signal_rates, noise_rates
            )

            # one of the losses is used for training, the others are tracked as metrics
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

        # track the exponential moving averages of weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        # KID is not measured during the training phase for computational efficiency
        return {m.name: m.result() for m in self.metrics[:-1]}

    def test_step(self, images):
        # normalize images to have standard deviation of 1, like the noises
        images = self.augmenter(images, training=False)
        noises = tf.random.normal(
            shape=(self.batch_size, self.image_size, self.image_size, 3)
        )

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(self.batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        signal_rates, noise_rates = self.diffusion_schedule(diffusion_times)

        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises
        velocities = -noise_rates * images + signal_rates * noises

        # use the network to separate noisy images to their components
        predictions = self.ema_network([noisy_images, noise_rates**2], training=False)
        pred_velocities, pred_images, pred_noises = self.get_components(
            noisy_images, predictions, signal_rates, noise_rates
        )

        velocity_loss = self.loss(velocities, pred_velocities)
        image_loss = self.loss(images, pred_images)
        noise_loss = self.loss(noises, pred_noises)

        self.velocity_loss_tracker.update_state(velocity_loss)
        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        # measure KID between real and generated images
        # this is computationally demanding, kid_diffusion_steps has to be small
        images = self.denormalize(images)
        generated_images = self.generate(
            self.batch_size,
            diffusion_steps=self.kid_diffusion_steps,
            stochasticity=0.0,
            variance_preserving=False,
            num_multisteps=1,
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
        stochasticity=0.0,
        variance_preserving=False,
        num_multisteps=1,
        second_order_alpha=None,
        plot_image_size=128,
    ):
        # plot random generated images for visual evaluation of generation quality
        generated_images = self.generate(
            num_rows * num_cols,
            diffusion_steps,
            stochasticity,
            variance_preserving,
            num_multisteps,
            second_order_alpha,
        )

        # organize generated images into a grid
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
        if self.is_jupyter:
            plt.figure(figsize=(num_cols * 1.5, num_rows * 1.5))
            plt.imshow(generated_images.numpy())
            plt.axis("off")
            plt.tight_layout()
            plt.show()
            plt.close()
        else:
            plt.imsave(
                "images/{}_e{}_s{:.1f}_k{:.3f}.png".format(
                    self.id,
                    "" if epoch is None else epoch + 1,
                    stochasticity,
                    self.kid.result(),
                ),
                generated_images.numpy(),
            )
