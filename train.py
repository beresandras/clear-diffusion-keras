import os
import matplotlib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # suppress info-level logs
matplotlib.use("Agg")

import tensorflow as tf
import tensorflow_addons as tfa

tf.get_logger().setLevel("WARN")  # suppress info-level logs

from tensorflow import keras

from dataset import prepare_dataset
from architecture import get_augmenter, get_network
from model import DiffusionModel


# hyperparameters

# data
# some datasets might be unavailable for download at times
dataset_name = "oxford_flowers102"
epochs = {
    "caltech_birds2011": 40,
    "oxford_flowers102": 40,
    "celeb_a": 20,
    "cifar10": 80,
}
num_epochs = epochs[dataset_name]
uncropped_image_size = 64
image_size = 64
kid_image_size = 75  # resolution of KID measurement (75/150/299)
kid_diffusion_steps = 5

# optimization
prediction_type = "noise"
loss_type = "noise"
batch_size = 64
ema = 0.999
learning_rate = 1e-3
weight_decay = 1e-4

# sampling
schedule_type = "cosine"
start_log_snr = 2.5
end_log_snr = -7.5

# architecture
noise_embedding_max_frequency = 200.0
noise_embedding_dims = 32
image_embedding_dims = 64
block_depth = 2
widths = [32, 64, 96, 128]
attentions = [False, False, False, False]

id = 0

# load dataset
train_dataset = prepare_dataset(dataset_name, "train", uncropped_image_size, batch_size)
val_dataset = prepare_dataset(
    dataset_name, "validation", uncropped_image_size, batch_size
)

# create model
model = DiffusionModel(
    id=id,
    augmenter=get_augmenter(
        uncropped_image_size=uncropped_image_size, image_size=image_size
    ),
    network=get_network(
        image_size=image_size,
        noise_embedding_max_frequency=noise_embedding_max_frequency,
        noise_embedding_dims=noise_embedding_dims,
        image_embedding_dims=image_embedding_dims,
        block_depth=block_depth,
        widths=widths,
        attentions=attentions,
    ),
    prediction_type=prediction_type,
    loss_type=loss_type,
    batch_size=batch_size,
    ema=ema,
    schedule_type=schedule_type,
    start_log_snr=start_log_snr,
    end_log_snr=end_log_snr,
    kid_image_size=kid_image_size,
    kid_diffusion_steps=kid_diffusion_steps,
)

model.compile(
    optimizer=tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    ),
    loss=keras.losses.mean_absolute_error,
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
model.plot_images(epoch=0)
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
