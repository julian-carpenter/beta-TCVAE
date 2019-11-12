import os
import pickle
from glob import glob
from numpy import log2, pi
import tensorflow as tf
# import tensorflow_addons as tfa

from nn.util import ResNetBlock, SelfAttentionModule

tfk = tf.keras
tfkl = tf.keras.layers


def get_models(args, log_dir=None):
    class betaTCVAE(tfk.Model):
        def __init__(self, args, log_dir=None):
            super(betaTCVAE, self).__init__()
            # MODEL
            if log_dir is not None:
                args_file = os.path.join(log_dir, "args.P")
            else:
                args_file = ""
            if os.path.isfile(args_file):
                with open(args_file, 'rb') as handle:
                    self.args = pickle.load(handle)
            else:
                with open(os.path.join(args.log_dir, "args.P"), 'wb') as handle:
                    pickle.dump(args, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    self.args = args

            self.args.logger.info("\t[*] Building Encoder")
            self.enc = encoder(self.args)
            self.args.logger.info("\t[*] Building Decoder")
            self.dec = decoder(self.args)

            self.args.logger.info("\t[*] Building Model")
            decoded = self.reparameterization_trick(self.enc.output)
            z = self.reparameterization_trick(self.enc.output, True)

            self.model = tfk.Model(inputs=self.enc.input,
                                   outputs=[decoded, z, self.encode(self.enc.output)])  # [decoded, z, Mean/LogVar]

            # ############## OPTIMIZER
            # We use RAdam with lookahead. Quite fancy stuff.
            # -> We use this only when tensorflow-addons get their shit done and compile tfa using the same cuda version
            # --> https://github.com/tensorflow/addons/issues/574

            # RAdam:
            # arXiv:1908.03265 [cs.LG]
            # Lookahead:
            # arXiv: 1907.08610[cs.LG]

            # radam = tfa.optimizers.RectifiedAdam(learning_rate=self.args.lr_vae,
            #                                      beta_1=self.args.beta1,
            #                                      beta_2=self.args.beta2,
            #                                      epsilon=self.args.epsilon)
            # self.opt_vae = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)

            # Use Adam with warmup.
            self.opt_vae = tfk.optimizers.Adam(learning_rate=self.args.lr_vae,
                                               beta_1=self.args.beta1,
                                               beta_2=self.args.beta2,
                                               epsilon=self.args.epsilon)

            # ############## LOADING
            if not os.path.isfile(args.use_specific_weight_file + ".index"):
                file_list = glob(os.path.join(self.args.log_dir, "cp-*.index"))
                self.args.logger.info("\t[*] Checking for trained model")
                if len(file_list) > 0:
                    file_list.sort()
                    latest = os.path.basename(file_list[-1])[:-6]
                    self.step = int(latest[-latest[::-1].find("-"):-latest[::-1].find(".") - 1])
                    self.args.logger.info("\t[*] Found: {}".format(latest))
                    self.model.load_weights(os.path.join(self.args.log_dir, latest))
                    self.args.logger.info("\t[*] Loading successful")
                else:
                    self.args.logger.info("\t[*] No model found. Starting from scratch.")
                    self.step = 0
                    # Save weights for epoch 0
                    ckpt_path_vae = os.path.join(self.args.log_dir, self.args.ckpt_name)
                    self.model.save_weights(ckpt_path_vae.format(epoch=0))
            else:
                self.model.load_weights(args.use_specific_weight_file)

        @tf.function
        def sample(self, eps=None):
            if eps is None:
                eps = tf.random.normal(shape=(100, self.args.z_dim))
            return self.decode(eps, apply_sigmoid=True)

        def reparameterization_trick(self, x, return_z=False):
            mean, log_squared_scale = self.encode(x)
            z = self.reparameterize(mean, log_squared_scale)
            if return_z:
                return z
            return self.decode(z)

        @staticmethod
        def encode(x):
            mean, log_squared_scale = tf.split(x, num_or_size_splits=[args.z_dim, args.z_dim], axis=1)
            return mean, log_squared_scale

        def reparameterize(self, mean, log_squared_scale):
            """Samples from the Gaussian distribution defined by z_mean and z_log_squared_scale."""
            if self.args.prior.lower() == "normal":
                return tf.math.add(
                    mean,
                    tf.math.exp(log_squared_scale / 2) * tf.random.normal(tf.shape(mean), 0, 1),
                    name="sampled_latent_variable")
            if self.args.prior.lower() == "laplace":
                lp_samples = self.lp_samples(tf.shape(mean))
                return tf.math.add(
                    mean,
                    tf.math.exp(log_squared_scale / 2) * lp_samples,
                    name="sampled_latent_variable")

        @staticmethod
        def lp_one_sided(s, u, b):
            return -tf.math.log(tf.random.uniform(s, u, b))

        def lp_samples(self, s, u=0, b=1):
            one_side = self.lp_one_sided(s, u, b)
            the_other_side = self.lp_one_sided(s, u, b)
            return one_side - the_other_side

        def decode(self, z, apply_sigmoid=False):
            z = tf.cast(z, tf.float32)
            logits = self.dec(z)
            if apply_sigmoid:
                probs = tf.math.sigmoid(logits)
                return probs

            return logits

    return betaTCVAE(args, log_dir)


def encoder(args):
    model = tfk.Sequential(name="Encoder")
    model.add(tfkl.InputLayer(input_shape=args.input_shape[1:]))
    model.add(tfkl.Lambda(lambda x: tf.cast(x, tf.float32) - 0.5))

    fs = int(args.img_size / 2 ** args.nr_conv)
    cn = log2(args.channels).astype(int)
    channels = [int(2 ** x) for x in range(cn, cn + args.nr_conv)]
    for ii, c in enumerate(channels):
        if args.use_residual_blocks and ii > 0:
            model.add(ResNetBlock(int(c), args, "down"))
        else:
            model.add(tfkl.Conv2D(c, 7 if ii == 0 else 4, strides=2, padding="same",
                                  activation=tf.nn.relu,
                                  kernel_initializer=args.initializer,
                                  kernel_regularizer=args.regularizer))
        if args.use_self_attention and ii == len(channels) - int(16 / fs):
            model.add(SelfAttentionModule(c, args))

    model.add(tfkl.Flatten())
    model.add(tfkl.Dense(256, activation=tf.nn.relu,
                         kernel_initializer=args.initializer,
                         kernel_regularizer=args.regularizer))
    # No activation
    model.add(tfkl.Dense(args.z_dim + args.z_dim,
                         kernel_initializer=args.initializer,
                         kernel_regularizer=args.regularizer))
    return model


def decoder(args):
    model = tfk.Sequential(name="Decoder")
    model.add(tfkl.InputLayer(input_shape=[args.z_dim], name="DecoderIn"))
    model.add(tfkl.Reshape([1, 1, args.z_dim]))

    # for n in nodes:
    fs = int(args.img_size / 2 ** args.nr_conv)
    model.add(tfkl.Dense(256, activation=tf.nn.relu,
                         kernel_initializer=args.initializer,
                         kernel_regularizer=args.regularizer))
    model.add(tfkl.Dense(int(fs ** 2 * args.channels), activation=tf.nn.relu,
                         kernel_initializer=args.initializer,
                         kernel_regularizer=args.regularizer))

    model.add(tfkl.Reshape([fs, fs, args.channels]))
    cn = log2(args.channels).astype(int)
    channels = [int(2 ** x) for x in range(cn + 1, cn + args.nr_conv)][::-1] + [args.img_channels]
    for ii, c in enumerate(channels):
        if args.use_self_attention and ii == int(16 / fs):
            model.add(SelfAttentionModule(channels[ii - 1], args))
        if args.use_residual_blocks and ii <= (len(channels) - 2):
            model.add(ResNetBlock(int(c) if c > 1 else 1, args, "up"))
        else:
            model.add(tfkl.Conv2DTranspose(c, 4 if ii < (len(channels) - 1) else 7,
                                           strides=2, padding="same",
                                           activation=tf.nn.relu if ii < (len(channels) - 1) else None,
                                           kernel_initializer=args.initializer,
                                           kernel_regularizer=args.regularizer))

    # No activation
    model.add(tfkl.Conv2DTranspose(filters=1, kernel_size=3, strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=args.initializer,
                                   kernel_regularizer=args.regularizer))

    return model
