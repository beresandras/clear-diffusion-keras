import tensorflow as tf

from abc import abstractmethod


class DiffusionSchedule:
    def __init__(self, start_log_snr, end_log_snr):
        assert (
            start_log_snr > end_log_snr
        ), "The starting SNR has to be higher than the final SNR."

        self.start_snr = tf.exp(start_log_snr)
        self.end_snr = tf.exp(end_log_snr)

        self.start_noise_power = 1.0 / (1.0 + self.start_snr)
        self.end_noise_power = 1.0 / (1.0 + self.end_snr)

    def __call__(self, diffusion_times):
        noise_powers = self.get_noise_powers(diffusion_times)

        # the signal and noise power will always sum to one
        signal_powers = 1.0 - noise_powers

        # the rates are the square roots of the powers
        # variance**0.5 -> standard deviation
        signal_rates = signal_powers**0.5
        noise_rates = noise_powers**0.5

        return signal_rates, noise_rates

    @abstractmethod
    def get_noise_powers(self, diffusion_times):
        pass


class LinearSchedule(DiffusionSchedule):
    # variance or power of noise component increases linearly
    def get_noise_powers(self, diffusion_times):
        return self.start_noise_power + diffusion_times * (
            self.end_noise_power - self.start_noise_power
        )


class CosineSchedule(DiffusionSchedule):
    # noise rate increases sinusoidally
    # signal rate decreases as a cosine function
    # simplified from the "cosine schedule" of Improved DDPM https://arxiv.org/abs/2102.09672
    def get_noise_powers(self, diffusion_times):
        start_angle = tf.asin(self.start_noise_power**0.5)
        end_angle = tf.asin(self.end_noise_power**0.5)

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
        return tf.sin(diffusion_angles) ** 2


class LogSNRLinearSchedule(DiffusionSchedule):
    # the log signal-to-noise ratio decreases linearly
    # proposed in VDM https://arxiv.org/abs/2107.00630
    def get_noise_powers(self, diffusion_times):
        return self.start_snr**diffusion_times / (
            self.start_snr * self.end_snr**diffusion_times
            + self.start_snr**diffusion_times
        )


class LogNoiseLinearSchedule(DiffusionSchedule):
    # the log noise power increases linearly
    # the noise power increases exponentially
    # the ratio between next-step and current noise powers is constant
    def get_noise_powers(self, diffusion_times):
        return (
            self.start_noise_power
            * (self.end_noise_power / self.start_noise_power) ** diffusion_times
        )


class LogSignalLinearSchedule(DiffusionSchedule):
    # the log signal power decreases linearly
    # the signal power decreases exponentially
    # the ratio between next-step and current signal powers is constant
    def get_noise_powers(self, diffusion_times):
        return (
            1.0
            - (1.0 - self.start_noise_power)
            * ((1.0 - self.end_noise_power) / (1.0 - self.start_noise_power))
            ** diffusion_times
        )


class NoiseStepLinearSchedule(DiffusionSchedule):
    # the ratio between next-step and current noise powers decreases approximately linearly to 1
    def get_noise_powers(self, diffusion_times):
        return self.end_noise_power * (
            self.start_noise_power / self.end_noise_power
        ) ** ((1.0 - diffusion_times) ** 2)


class SignalStepLinearSchedule(DiffusionSchedule):
    # the ratio between next-step and current signal powers decreases approximately linearly to 1
    # similar to the "linear schedule" of DDPM https://arxiv.org/abs/2006.11239
    def get_noise_powers(self, diffusion_times):
        return 1.0 - (1.0 - self.start_noise_power) * (
            (1.0 - self.end_noise_power) / (1.0 - self.start_noise_power)
        ) ** (diffusion_times**2)
