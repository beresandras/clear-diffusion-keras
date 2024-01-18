import os
import matplotlib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # suppress info-level logs
matplotlib.use("Agg")

import tensorflow as tf

tf.get_logger().setLevel("WARN")  # suppress info-level logs

from tensorflow import keras

from dataset import (
    BirdsDataset,
    ButterfliesMuseumDataset,
    ButterfliesNatureDataset,
    CelebsDataset,
    FlowersDataset,
    PokemonsDataset,
)
from architecture import get_augmenter, get_network
from schedule import SignalStepLinearSchedule, LogSNRLinearSchedule
from model import DiffusionModel


# hyperparameters

# optimization
prediction_type = "velocity"
loss_type = "velocity"
batch_size = 64
ema = 0.999
learning_rate = 2e-4
weight_decay = 1e-4

# data
# some datasets might be unavailable for download at times
num_epochs = 40
image_size = 64
dataset = FlowersDataset(image_size=image_size, batch_size=batch_size)

# sampling
diffusion_schedule = SignalStepLinearSchedule(
    start_log_snr=3.0,
    end_log_snr=-10.0,
)
kid_image_size = 75  # resolution of KID measurement (75/150/299)
kid_diffusion_steps = 5

# architecture
noise_embedding_max_frequency = 1000.0
noise_embedding_dims = 64
image_embedding_dims = 64
block_depth = 2
widths = [64, 128, 256, 512]  # smaller: [32, 64, 96, 128]
attentions = [False, False, True, True]  # smaller: [False, False, False, False]
patch_size = 1

id = 0

# load dataset
train_dataset = dataset.to_tf_dataset(split="train")
val_dataset = dataset.to_tf_dataset(split="validation")

# create model
model = DiffusionModel(
    id=id,
    augmenter=get_augmenter(image_size=image_size),
    network=get_network(
        image_size=image_size,
        noise_embedding_max_frequency=noise_embedding_max_frequency,
        noise_embedding_dims=noise_embedding_dims,
        image_embedding_dims=image_embedding_dims,
        block_depth=block_depth,
        widths=widths,
        attentions=attentions,
        patch_size=patch_size,
    ),
    prediction_type=prediction_type,
    loss_type=loss_type,
    batch_size=batch_size,
    ema=ema,
    diffusion_schedule=diffusion_schedule,
    kid_image_size=kid_image_size,
    kid_diffusion_steps=kid_diffusion_steps,
)

model.compile(
    optimizer=keras.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    ),
    loss=keras.losses.mean_squared_error,
)

# checkpointing
checkpoint_path = "checkpoints/model_{}".format(id)
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor="val_kid",
    mode="min",
    save_best_only=True,
)

# run training
model.augmenter.layers[0].adapt(train_dataset)  # normalize images
model.plot_images(epoch=-1)
model.fit(
    train_dataset,
    epochs=num_epochs,
    validation_data=val_dataset,
    callbacks=[
        keras.callbacks.LambdaCallback(on_epoch_end=model.plot_images),
        checkpoint_callback,
    ],
)

# load best model
model.load_weights(checkpoint_path)
# model.plot_images(diffusion_steps=20, seed=42)
# model.evaluate(val_dataset)

# DDIM sampling
model.plot_images(diffusion_steps=20)

# DDIM multistep sampling
model.plot_images(diffusion_steps=20, num_multisteps=2)

# DDIM second order sampling
model.plot_images(diffusion_steps=20, second_order_alpha=0.5)

# DDPM variance preserving sampling
model.plot_images(diffusion_steps=20, stochasticity=1.0, variance_preserving=True)

# DDPM sampling with large variance
model.plot_images(diffusion_steps=200, stochasticity=1.0, variance_preserving=False)
