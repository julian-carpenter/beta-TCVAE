"""
This is an implementation of the beta-TCVAE

See:
    Ricky T. Q. Chen, Xuechen Li, Roger Grosse, David Duvenaud:
    Isolating Sources of Disentanglement in Variational Autoencoders
    arXiv:1802.04942 [cs.LG]
"""
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import os
    import logging
    import numpy as np
    from tqdm import tqdm
    import tensorflow as tf

    from nn import get_args
    from nn.data import get_data
    from nn.models import get_models
    from nn.losses import ae_reconstruction_loss, kl_penalty, tc_penalty
    from nn.util import rate_scheduler, auto_size
    from nn.evaluation import eval_model

# SETTING GLOBAL VARIABLES
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_FLAGS"] = "--xla_hlo_profile"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"

tfk = tf.keras
tfkl = tf.keras.layers


def train(args):
    max_mis = 0.0
    if args.batch_size == -1:
        # find the best batch_size, such that the modulus of the batch_size and the
        # sample_size is the largest '0' between 50 and 100.
        args.batch_size = auto_size(50, 100, args.sample_size)
        args.logger.info("\t[*] Setting batch_size to: {}".format(args.batch_size))

    args.steps_per_epoch = int(args.sample_size /
                               args.batch_size)

    if args.log_steps == -1:
        # find the best log_step, such that the modulus of the log_step and the
        # steps_per_epoch is the largest '0' between 20 and 100.
        args.log_steps = auto_size(20, 100, args.steps_per_epoch)
        args.logger.info("\t[*] Setting log_step to: {}".format(args.log_steps))

    args.global_step = 0
    args.ckpt_name = "cp-{epoch:06d}.ckpt"
    args.z_dim = int(args.z_dim)
    args.nr_conv = 5  # Number of conv / trans. conv layers
    args.channels = 32

    # ############## THE DATA
    if args.dataset.lower() in ["dynamic_helium", "helium", "mnist", "cifar10", "fashion_mnist"]:
        with tf.device('/cpu:0'):  # Do the sampling on the CPU
            data = get_data(args)
    else:
        args.logger.info("Select a valid dataset. "
                         "Valid choices are: 'helium' and 'mnist'")
        raise FileNotFoundError

    args.input_shape = [args.batch_size,
                        args.img_size,
                        args.img_size,
                        args.img_channels]
    args.logger.info("\t[*] Input Dimension: {}".format(args.input_shape))

    # ############## THE MODEL
    args.initializer = "glorot_normal"
    args.regularizer = tfk.regularizers.l2(args.weight_decay)

    betaTCVAE = get_models(args)

    args.logger.info("\t[*] Encoder summary:")
    betaTCVAE.enc.summary(print_fn=args.logger.info)
    args.logger.info("\t[*] Decoder summary:")
    betaTCVAE.dec.summary(print_fn=args.logger.info)

    log_writer = tf.summary.create_file_writer(args.log_dir)

    # Build lr scheduler -> linearly go to zero, after some warmup
    if args.lr_to_zero:
        duration_training = int(args.epochs * args.steps_per_epoch - args.num_warmup_steps)

        def lin_fun(x):
            return np.hstack([np.ones(args.num_warmup_steps) * x, np.linspace(x, 0, duration_training)])

        learning_rates_vae = (x for x in lin_fun(args.lr_vae))

    # Train the model
    e__ = 0 + int(betaTCVAE.step / args.steps_per_epoch)
    with log_writer.as_default():
        it_ = tqdm(data, total=args.steps_per_epoch * args.epochs)

        for x, lbl in it_:
            with tf.GradientTape() as vae_tape:
                # Sample VAE
                logits, z, (mean, log_var) = betaTCVAE.model(x)

                # Train VAE
                args.annealed_beta = rate_scheduler(args.global_step,
                                                    int(args.steps_per_epoch * args.epochs / 2),
                                                    args.beta,
                                                    args.beta) + 1.
                ae_loss = ae_reconstruction_loss(x, logits, args.true_images)
                kl_loss = kl_penalty(mean, log_var, args.prior)
                tc_loss, tc = tc_penalty(args, betaTCVAE.reparameterize(mean, log_var),
                                         mean, log_var, args.prior)

                elbo = tf.math.add(ae_loss, kl_loss, name="elbo")
                loss = tf.math.add(elbo, tc_loss, name="loss")

                vae_grads = vae_tape.gradient(loss, betaTCVAE.model.trainable_weights)
                betaTCVAE.opt_vae.apply_gradients(zip(vae_grads, betaTCVAE.model.trainable_weights))

                tf.debugging.check_numerics(kl_loss, "KL is not numeric")
                tf.debugging.check_numerics(tc_loss, "TC is not numeric")

                # Logging
                losses = [x.numpy().mean() for x in [loss, -elbo, ae_loss, kl_loss, -tc_loss, -tc]]
                loss_names = ["loss", "elbo", "reconstruction", "kl-penalty", "tc-penalty", "tc-estimate"]

                if (args.global_step + betaTCVAE.step) % args.log_steps == 0:
                    for l, n in zip(losses, loss_names):
                        tf.summary.scalar("losses/{}".format(n), l,
                                          step=args.global_step + betaTCVAE.step)
                    tf.summary.image(
                        "Original",
                        x,
                        step=args.global_step + betaTCVAE.step,
                        max_outputs=2
                    )
                    tf.summary.image(
                        "Generated",
                        tf.math.sigmoid(logits),
                        step=args.global_step + betaTCVAE.step,
                        max_outputs=2
                    )

                str_list = ["{}: {:02.02e} ".format(a, b) for a, b in zip(loss_names, losses)]
                loss_str = "".join(str_list) + "| Epoch: {}".format(e__)
                it_.set_postfix_str(loss_str)

                # Adjusting learning rate
                new_vae_lr = rate_scheduler(args.global_step,
                                            args.num_warmup_steps,
                                            args.lr_vae,
                                            next(learning_rates_vae) if args.lr_to_zero else args.lr_vae)
                betaTCVAE.opt_vae.lr.assign(new_vae_lr)

                assert np.allclose(betaTCVAE.opt_vae.lr.read_value(), new_vae_lr)

                if (args.global_step + betaTCVAE.step) % args.log_steps == 0:
                    tf.summary.scalar("model/vae_lr", betaTCVAE.opt_vae.lr.read_value(),
                                      step=args.global_step + betaTCVAE.step)

            if (args.global_step + betaTCVAE.step) % args.steps_per_epoch == 0:  # reached an epoch

                ckpt_path_vae = os.path.join(args.log_dir, args.ckpt_name)
                betaTCVAE.model.save_weights(
                    ckpt_path_vae.format(epoch=int(args.global_step + betaTCVAE.step)))

                if e__ % args.eval_epochs == 0 and e__ > 0:
                    args.logger.info("\t[*] Starting eval")

                    with tf.device("/cpu:0"):
                        eval_model(args, betaTCVAE, epoch=e__)
                e__ += 1

            args.global_step += 1

    # Save the whole model after training finished
    finished_model_path = os.path.join(
        args.result_dir,
        "trained_for_{}_epochs".format(args.epochs))
    betaTCVAE.model.save_weights(finished_model_path)


def main(args):
    # parse arguments
    if args is None:
        exit()

    # Build the model train it
    train(args)


if __name__ == "__main__":
    args = get_args()

    # tf.config.set_soft_device_placement(True)
    args.logger = tf.get_logger()
    args.logger.setLevel(logging.INFO)
    args.logger.propagate = False

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(args.log_dir, "train.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # args.logger.addHandler(ch)
    args.logger.addHandler(fh)
    has_gpu = tf.test.is_gpu_available()

    if has_gpu:
        main(args)

        # close the handlers
        for handler in args.logger.handlers:
            handler.close()
            args.logger.removeFilter(handler)
    else:
        args.logger.info("No GPU, No VAE")
