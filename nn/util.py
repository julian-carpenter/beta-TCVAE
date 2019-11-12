"""
Useful utility functions for betaTCVAE
"""
import os

import numba
import numpy as np
import tensorflow as tf
from scipy import ndimage

tfk = tf.keras
tfkl = tf.keras.layers


class SelfAttentionModule(tf.keras.Model):
    """
    Self-attention module composed of convolutional layers.

    See here:
    1) Self-Attention Generative Adversarial Networks (arXiv:1805.08318)

    """

    def __init__(self,
                 attention_features,
                 args):
        """
        Initialize the module.

        Args:
          attention_features: Number of filters for the attention computation.
          args: Arguments from runtime
        """
        super(SelfAttentionModule, self).__init__()
        # Matrix multiplication implemented as 2D Convolution
        self.f = tfkl.Conv2D(attention_features, 1,
                             kernel_initializer=args.initializer,
                             kernel_regularizer=args.regularizer)
        self.g = tfkl.Conv2D(attention_features, 1,
                             kernel_initializer=args.initializer,
                             kernel_regularizer=args.regularizer)
        self.h = tfkl.Conv2D(attention_features, 1,
                             kernel_initializer=args.initializer,
                             kernel_regularizer=args.regularizer)
        self.scale = tf.Variable(0., trainable=True)
        self.args = args

    def call(self, x):
        f = self.f(x)
        g = self.g(x)
        h = self.h(x)

        f_flatten = flatten_hw(f)
        g_flatten = flatten_hw(g)
        h_flatten = flatten_hw(h)

        s = tf.math.matmul(g_flatten, f_flatten, transpose_b=True)
        b = tf.math.softmax(s, axis=-1)
        o = tf.math.matmul(b, h_flatten)
        y = self.scale * tf.reshape(o, tf.shape(x)) + x

        tf.summary.scalar("attention_scale", self.scale,
                          step=self.args.global_step)
        # print("SA: In: {} Out: {}".format(x.shape, y.shape))
        return y


class ResNetBlock(tf.keras.Model):
    def __init__(self, size, args, resampling="id"):
        """
        Implements a Basic Pre-Activated Residual Block.

        See here:
        1) Deep Residual Learning for Image Recognition (arXiv:1512.03385)
        2) Identity Mappings in Deep Residual Networks (arXiv:1603.05027)

        :param size: Filter size for the convolutions
        :param args: Runtime Arguments
        :param resampling: Must be: 1) 'id'  : Identity ResNetBlock
                                    2) 'up'  : Upsampling ResNetBlock (Using transposed conv's)
                                    3) 'down': Downsampling ResNetBlock
        """
        super(ResNetBlock, self).__init__(name='')
        self.args = args
        if resampling == "id":
            s = 1
        else:
            s = 2

        if resampling == "up":
            self.sample_op = tfkl.Conv2DTranspose(size, 1, s, kernel_initializer=self.args.initializer,
                                                  kernel_regularizer=self.args.regularizer, padding="valid")
            self.conv3x3_1 = tfkl.Conv2DTranspose(size, 3, s, kernel_initializer=self.args.initializer,
                                                  kernel_regularizer=self.args.regularizer, padding="same")
        else:
            if resampling == "down":
                self.sample_op = tfkl.Conv2D(size, 1, s, kernel_initializer=self.args.initializer,
                                             kernel_regularizer=self.args.regularizer, padding="valid")
            else:  # resampling=="id"
                self.sample_op = tfkl.Lambda(lambda x: x)
            self.conv3x3_1 = tfkl.Conv2D(size, 3, s, kernel_initializer=self.args.initializer,
                                         kernel_regularizer=self.args.regularizer, padding="same")

        self.bn_1 = tfkl.BatchNormalization()

        self.conv3x3_2 = tfkl.Conv2D(size, 3, kernel_initializer=self.args.initializer,
                                     kernel_regularizer=self.args.regularizer, padding="same")
        self.bn_2 = tfkl.BatchNormalization()

    def call(self, x_in, training=False):

        x = self.conv3x3_1(x_in)
        x = self.bn_1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv3x3_2(x)
        x = self.bn_2(x, training=training)
        # x = tf.nn.relu(x)

        x += self.sample_op(x_in)
        # print("RES: In: {} Out: {}". format(x_in.shape, x.shape))
        return x


def flatten_hw(x):
    """Flatten the input tensor across height and width dimensions."""
    old_shape = tf.shape(x)
    new_shape = [old_shape[0], old_shape[2] * old_shape[3], old_shape[1]]

    return tf.reshape(x, new_shape)


def str2bool(x):
    return x.lower() in "true"


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def norm_img(data, min_new=0, max_new=1):
    """Normalize data to max_new and  min_new."""
    if min_new != max_new:
        data = tf.dtypes.cast(data, dtype=tf.float32)
        min_new = tf.dtypes.cast(min_new, dtype=tf.float32)
        max_new = tf.dtypes.cast(max_new, dtype=tf.float32)
        max_old = tf.math.reduce_max(data)
        min_old = tf.math.reduce_min(data)

        quot = tf.math.divide(tf.math.subtract(max_new, min_new),
                              tf.math.subtract(max_old, min_old))

        mult = tf.math.multiply(quot, tf.math.subtract(data, max_old))
        data = tf.math.add(mult, max_new)
        return data
    else:
        raise ArithmeticError


def rate_scheduler(global_step, num_warmup_steps, init_lr, learning_rate):
    global_steps_int = tf.dtypes.cast(global_step, tf.int32)
    warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

    global_steps_float = tf.dtypes.cast(global_steps_int, tf.float32)
    warmup_steps_float = tf.dtypes.cast(warmup_steps_int, tf.float32)

    warmup_percent_done = global_steps_float / warmup_steps_float
    warmup_learning_rate = init_lr * warmup_percent_done

    is_warmup = tf.dtypes.cast(global_steps_int < warmup_steps_int, tf.float32)
    return (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate


def auto_size(valid_range_min, valid_range_max, mult):
    valid_range = np.arange(valid_range_min, valid_range_max)
    mods = mult % valid_range
    return valid_range[::-1][np.argmin(mods[::-1])]
