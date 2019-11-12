import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import random_rotation


def rot(img):
    return random_rotation(np.array(img), 90, row_axis=0, col_axis=1, channel_axis=2)


def dataset_parse_function(x, args):
    """
    A parser for a tf dataset.
    Please note that the helium dataset is inhouse and not publicly available.
    :param x:
    :param args:
    :return:
    """
    if "helium" in args.dataset.lower():
        img = tf.io.decode_raw(x["imgs"], out_type=tf.float32)
        img = tf.reshape(img, [args.real_img_size, args.real_img_size, args.img_channels])
    else:
        img = x["image"]
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)  # Scale to unit interval.

    if "helium" in args.dataset.lower():
        if args.split == "train":
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            img = tf.py_function(func=rot, inp=[img], Tout=tf.float32)  # TODO: Replace with faster alternative
            img = tf.reshape(img, [args.real_img_size, args.real_img_size, args.img_channels])
            k = tf.random.uniform([1],
                                  minval=0,
                                  maxval=4,
                                  dtype=tf.dtypes.int32,
                                  seed=1612)
            img = tf.image.rot90(img, k=tf.squeeze(k))

        if args.use_central_cutoff:
            # This cuts out the central part of the image so that
            # the height and width of the cropped area are fully
            # within the diffraction pattern.
            img = tf.image.central_crop(img, np.cos(np.pi / 4))

    if args.real_img_size != args.img_size:
        method = tf.compat.v2.image.ResizeMethod.LANCZOS5
        img = tf.compat.v2.image.resize(img,
                                        [args.img_size] * 2,
                                        method=method)

    if args.dataset.lower() == "helium":
        lbl = x["lbl"]
        if args.split == "test":
            rad = x["radii"]
            return img, lbl, rad
        return img, lbl

    if args.dataset.lower() == "dynamic_helium":
        bid = x["bid"]  # no label information available, pass on BunchID as a unique identifier
        return img, bid

    if np.shape(args.dataset_info.features["image"])[-1] != 1:
        img = tf.image.rgb_to_grayscale(img)

    lbl = x["label"]
    return img, (img, lbl)


def binarize_images(img, *args):
    img = img < tf.random.uniform(tf.shape(img))  # Randomly binarize.

    return [img] + list(args)


def get_data(args, split="train"):
    """
    Create and return a tf dataset

    Returns:
        A tf.data.dataset object
    """
    args.split = split
    if args.dataset.lower() == "helium":
        if split == "train":
            args.data_file = os.path.join(args.data_dir, "balanced_helium_with_radius.tfrecords")
        if split == "test":
            args.data_file = os.path.join(args.data_dir, "balanced_helium_with_radius.tfrecords")

        if not os.path.isfile(args.data_file):
            print("\t[!] No file found at: {}".format(args.data_dir), flush=True)
            raise FileNotFoundError
        else:
            print("\t[*] Input file(s): {}".format(args.data_file), flush=True)

        image_feature_description = {
            'imgs': tf.io.FixedLenFeature([], tf.string),
            'lbl': tf.io.FixedLenFeature([], tf.int64),
            'radii': tf.io.FixedLenFeature([], tf.float32),
        }

    if args.dataset.lower() == "dynamic_helium":
        args.data_file = os.path.join(args.data_dir, "dynamic_helium_non_spherical.tfrecords")

        if not os.path.isfile(args.data_file):
            print("\t[!] No file found at: {}".format(args.data_dir), flush=True)
            raise FileNotFoundError
        else:
            print("\t[*] Input file(s): {}".format(args.data_file), flush=True)

        # Create a dictionary describing the features.
        image_feature_description = {
            'imgs': tf.io.FixedLenFeature([], tf.string),
            'bid': tf.io.FixedLenFeature([], tf.int64),
        }

    if "helium" in args.dataset.lower():
        dataset = tf.data.TFRecordDataset(args.data_file)

        def _parse_image_function(example_proto):
            # Parse the input tf.Example proto using the dictionary above.
            return tf.io.parse_single_example(example_proto, image_feature_description)

        dataset = dataset.map(_parse_image_function)
        args.real_img_size = 256

    elif args.dataset.lower() == "mnist":
        dataset, args.dataset_info = tfds.load(name="mnist", split=args.split, with_info=True)
        args.real_img_size = args.dataset_info.features["image"].shape[0]
    elif args.dataset.lower() == "fashion_mnist":
        dataset, args.dataset_info = tfds.load(name="fashion_mnist", split=args.split, with_info=True)
        args.real_img_size = args.dataset_info.features["image"].shape[0]
    elif args.dataset.lower() == "cifar10":
        dataset, args.dataset_info = tfds.load(name="cifar10", split=args.split, with_info=True)
        args.real_img_size = args.dataset_info.features["image"].shape[0]
    else:
        print("Select a valid dataset. Valid choices are: 'helium', "
              "'cifar10', 'fashion_mnist' and 'mnist'", flush=True)
        raise IndexError

    dataset = dataset.map(lambda x: dataset_parse_function(x, args))
    if split == "train":
        dataset = dataset.shuffle(int(args.sample_size * 1.5))
        dataset = dataset.repeat(args.epochs)  # repetition
    if split == "test":
        dataset = dataset.shuffle(int(args.sample_size))
        dataset = dataset.repeat(1)  # no repetition

    if not args.true_images:
        dataset = dataset.map(binarize_images)

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(args.batch_size)  # set batch size
    return dataset
