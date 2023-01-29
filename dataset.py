from typing import Dict
import tensorflow as tf
import tensorflow_datasets as tfds
import datasets as hfds
from PIL import ImageOps

from abc import ABC, abstractmethod


def round_to_int(float_value):
    return tf.cast(tf.math.round(float_value), dtype=tf.int32)


class Dataset(ABC):
    name: str
    split_names: Dict[str, str]
    repetitions: Dict[str, int]

    def __init__(self, image_size, batch_size):
        self.image_size = image_size
        self.batch_size = batch_size

    @abstractmethod
    def preprocess(self, data):
        pass

    @abstractmethod
    def to_tf_dataset(self, split):
        pass


class TFDataset(Dataset):
    def to_tf_dataset(self, split):
        return (
            tfds.load(self.name, split=self.split_names[split], shuffle_files=True)
            .map(self.preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            .cache()
            .repeat(self.repetitions[split])
            .shuffle(10 * self.batch_size)
            .batch(self.batch_size, drop_remainder=True)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )


class HFDataset(Dataset):
    def to_tf_dataset(self, split):
        return (
            hfds.load_dataset(self.name, split=self.split_names[split])
            .map(self.preprocess)
            .to_tf_dataset(batch_size=1, columns="image", prefetch=False)
            .map(
                lambda x: tf.cast(x[0], dtype=tf.float32) / 255.0,
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .repeat(self.repetitions[split])
            .shuffle(10 * self.batch_size)
            .batch(self.batch_size, drop_remainder=True)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )


class BirdsDataset(TFDataset):
    name = "caltech_birds2011"
    split_names = {"train": "train", "validation": "test"}
    repetitions = {"train": 10, "validation": 2}
    padding = 0.25

    def preprocess(self, data):
        # unnormalize bounding box coordinates
        height = tf.cast(tf.shape(data["image"])[0], dtype=tf.float32)
        width = tf.cast(tf.shape(data["image"])[1], dtype=tf.float32)
        bounding_box = data["bbox"] * tf.stack([height, width, height, width])

        # calculate center, length of longer side, add padding
        target_center_y = 0.5 * (bounding_box[0] + bounding_box[2])
        target_center_x = 0.5 * (bounding_box[1] + bounding_box[3])
        target_size = tf.maximum(
            (1.0 + self.padding) * (bounding_box[2] - bounding_box[0]),
            (1.0 + self.padding) * (bounding_box[3] - bounding_box[1]),
        )

        # modify bounding box to fit into image
        target_height = tf.reduce_min(
            [target_size, 2.0 * target_center_y, 2.0 * (height - target_center_y)]
        )
        target_width = tf.reduce_min(
            [target_size, 2.0 * target_center_x, 2.0 * (width - target_center_x)]
        )

        # crop image
        image = tf.image.crop_to_bounding_box(
            data["image"],
            offset_height=round_to_int(target_center_y - 0.5 * target_height),
            offset_width=round_to_int(target_center_x - 0.5 * target_width),
            target_height=round_to_int(target_height),
            target_width=round_to_int(target_width),
        )

        # resize and clip
        image = tf.image.resize(
            image,
            size=[self.image_size, self.image_size],
            method="bicubic",
            antialias=True,
        )
        return tf.clip_by_value(image / 255.0, 0.0, 1.0)


class FlowersDataset(TFDataset):
    name = "oxford_flowers102"
    split_names = {
        "train": "train[:80%]+validation[:80%]+test[:80%]",
        "validation": "train[80%:]+validation[80%:]+test[80%:]",
    }
    repetitions = {"train": 10, "validation": 10}

    def preprocess(self, data):
        # center crop image
        height = tf.shape(data["image"])[0]
        width = tf.shape(data["image"])[1]
        crop_size = tf.minimum(height, width)
        image = tf.image.crop_to_bounding_box(
            data["image"],
            (height - crop_size) // 2,
            (width - crop_size) // 2,
            crop_size,
            crop_size,
        )

        # resize and clip
        image = tf.image.resize(
            image,
            size=[self.image_size, self.image_size],
            method="bicubic",
            antialias=True,
        )
        return tf.clip_by_value(image / 255.0, 0.0, 1.0)


class CelebsDataset(TFDataset):
    name = "celeb_a"
    split_names = {
        "train": "train",
        "validation": "validation",
    }
    repetitions = {"train": 1, "validation": 1}
    crop_size = 140

    def preprocess(self, data):
        # center crop image
        height = 218
        width = 178
        image = tf.image.crop_to_bounding_box(
            data["image"],
            (height - self.crop_size) // 2,
            (width - self.crop_size) // 2,
            self.crop_size,
            self.crop_size,
        )

        # resize and clip
        image = tf.image.resize(
            image,
            size=[self.image_size, self.image_size],
            method="bicubic",
            antialias=True,
        )
        return tf.clip_by_value(image / 255.0, 0.0, 1.0)


class CIFAR10Dataset(TFDataset):
    name = "cifar10"
    split_names = {"train": "train", "validation": "test"}
    repetitions = {"train": 1, "validation": 1}

    def preprocess(self, data):
        # no antialias, since we always upsample from 32x32
        image = tf.image.resize(
            data["image"], size=[self.image_size, self.image_size], method="bicubic"
        )
        return tf.clip_by_value(image / 255.0, 0.0, 1.0)


class ButterfliesMuseumDataset(HFDataset):
    name = "huggan/smithsonian_butterflies_subset"
    split_names = {"train": "train", "validation": "train"}
    repetitions = {"train": 50, "validation": 10}

    def preprocess(self, data):
        # center pad
        return {
            "image": ImageOps.pad(
                image=data["image"],
                size=(self.image_size, self.image_size),
                color="white",
            )
        }


class ButterfliesNatureDataset(HFDataset):
    name = "huggan/inat_butterflies_top10k"
    split_names = {"train": "train", "validation": "train"}
    repetitions = {"train": 5, "validation": 1}

    def preprocess(self, data):
        # center crop
        image = data["image"]
        width, height = image.size
        new_size = min(width, height)
        left = (width - new_size) // 2
        top = (height - new_size) // 2
        return {
            "image": image.resize(
                size=(self.image_size, self.image_size),
                box=(left, top, left + new_size, top + new_size),
            )
        }


class PokemonsDataset(HFDataset):
    name = "huggan/pokemon"
    split_names = {"train": "train", "validation": "train"}
    repetitions = {"train": 10, "validation": 1}

    def preprocess(self, data):
        # center pad
        return {
            "image": ImageOps.pad(
                image=data["image"],
                size=(self.image_size, self.image_size),
                color="white",
            )
        }
