import os
import matplotlib

from model import DiffusionModel

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # suppress info-level logs
import tensorflow as tf

from tensorflow import keras

from dataset import prepare_dataset
from architecture import get_augmenter, get_network

tf.get_logger().setLevel("WARN")  # suppress info-level logs

matplotlib.use("Agg")

# hyperparameters

# data
# some datasets might be unavailable for download at times
dataset_name = "oxford_flowers102"
image_size = 32
num_epochs = 10
kid_image_size = 75  # resolution of KID measurement, default 299
plot_interval = 1

# optimization
batch_size = 64
time_margin = 0.05
ema = 0.99
learning_rate = 1e-3
# weight_decay = 1e-4

# architecture
num_resolutions = 3
block_depth = 2
width = 32

id = 0

# load dataset
train_dataset = prepare_dataset(dataset_name, "train", image_size, batch_size)
val_dataset = prepare_dataset(dataset_name, "validation", image_size, batch_size)

# create model
model = DiffusionModel(
    augmenter=get_augmenter(image_size=image_size),
    network=get_network(
        image_size=image_size,
        num_resolutions=num_resolutions,
        block_depth=block_depth,
        width=width,
    ),
    batch_size=batch_size,
    time_margin=time_margin,
    ema=ema,
    kid_image_size=kid_image_size,
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    loss=keras.losses.mean_absolute_error,
)

# checkpointing
checkpoint_path = "checkpoints/model_{}".format(id)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor="val_kid",
    mode="min",
    save_best_only=True,
)

# run training
history = model.fit(
    train_dataset,
    epochs=num_epochs,
    validation_data=val_dataset,
    callbacks=[
        keras.callbacks.LambdaCallback(on_epoch_end=model.plot_images),
        checkpoint_callback,
    ],
)

# load best model
# model.load_weights(checkpoint_path)
# generate_images_with(model, history, id)

# plot history
# plot_history(history, id)