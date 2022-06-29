import tensorflow as tf
import tensorflow_datasets as tfds


def round_to_int(float_value):
    return tf.cast(tf.math.round(float_value), dtype=tf.int32)


def preprocess_birds(image_size, padding=0.25):
    def preprocess_image(data):
        # unnormalize bounding box coordinates
        height = tf.cast(tf.shape(data["image"])[0], dtype=tf.float32)
        width = tf.cast(tf.shape(data["image"])[1], dtype=tf.float32)
        bounding_box = data["bbox"] * tf.stack([height, width, height, width])

        # calculate center, length of longer side, add padding
        target_center_y = 0.5 * (bounding_box[0] + bounding_box[2])
        target_center_x = 0.5 * (bounding_box[1] + bounding_box[3])
        target_size = tf.maximum(
            (1.0 + padding) * (bounding_box[2] - bounding_box[0]),
            (1.0 + padding) * (bounding_box[3] - bounding_box[1]),
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
            image, size=[image_size, image_size], method="bicubic", antialias=True
        )
        return tf.clip_by_value(image / 255.0, 0.0, 1.0)

    return preprocess_image


def preprocess_flowers(image_size):
    def preprocess_image(data):
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
            image, size=[image_size, image_size], method="bicubic", antialias=True
        )
        return tf.clip_by_value(image / 255.0, 0.0, 1.0)

    return preprocess_image


def preprocess_celeba(image_size, crop_size=140):
    def preprocess_image(data):
        # center crop image
        height = 218
        width = 178
        image = tf.image.crop_to_bounding_box(
            data["image"],
            (height - crop_size) // 2,
            (width - crop_size) // 2,
            crop_size,
            crop_size,
        )

        # resize and clip
        image = tf.image.resize(
            image, size=[image_size, image_size], method="bicubic", antialias=True
        )
        return tf.clip_by_value(image / 255.0, 0.0, 1.0)

    return preprocess_image


def preprocess_cifar(image_size):
    def preprocess_image(data):
        # will always have a resolution of 32x32
        return tf.image.convert_image_dtype(data["image"], tf.float32)

    return preprocess_image


def prepare_dataset(dataset_name, split, image_size, batch_size):
    preprocessors = {
        "caltech_birds2011": preprocess_birds,
        "oxford_flowers102": preprocess_flowers,
        "celeb_a": preprocess_celeba,
        "cifar10": preprocess_cifar,
    }
    preprocess_image = preprocessors[dataset_name](image_size)

    split_index = {"train": 0, "validation": 1}
    split_names = {
        "caltech_birds2011": ["train", "test"],
        "oxford_flowers102": [
            "train[:80%]+validation[:80%]+test[:80%]",
            "train[80%:]+validation[80%:]+test[80%:]",
        ],
        "celeb_a": ["train", "validation"],
        "cifar10": ["train", "test"],
    }
    split_name = split_names[dataset_name][split_index[split]]

    repetitions = {
        "caltech_birds2011": [10, 2],
        "oxford_flowers102": [10, 10],
        "celeb_a": [1, 1],
        "cifar10": [1, 1],
    }
    repetition = repetitions[dataset_name][split_index[split]]

    # the validation dataset is shuffled as well, because data order matters
    # for the KID calculation
    return (
        tfds.load(dataset_name, split=split_name, shuffle_files=True)
        .map(
            preprocess_image,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .cache()
        .repeat(repetition)
        .shuffle(10 * batch_size)
        .batch(batch_size, drop_remainder=True)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )