import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

tfk = tf.keras


def kl_penalty(z_mean, z_log_squared_scale, prior="normal"):
    """
    Compute KL divergence between input dist and Standard dist (loc=0, scale=1).
    For normal:
    This is the first term in Eq. 10 from the seminal VAE paper:

    Diederik P Kingma, Max Welling: Auto-Encoding Variational Bayes
    arXiv:1312.6114 [stat.ML]

    For laplace:
    See: Gil, M., Alajaji, et al.
    Rényi divergence measures for commonly used univariate continuous distributions.
    Inf. Sci. (Ny). 249, 124–131 (2013).
    """
    if prior.lower() == "normal":
        summand = tf.math.square(z_mean) + tf.math.exp(z_log_squared_scale) - z_log_squared_scale - 1
        return tf.math.reduce_mean(0.5 * tf.math.reduce_sum(summand, [1]), name="kl_loss")
    if prior.lower() == "laplace":
        exponent = 0.5 * z_log_squared_scale - tf.math.abs(z_mean) * tf.math.exp(- 0.5 * z_log_squared_scale)
        summand = tf.math.abs(z_mean) + tf.math.exp(exponent) - 0.5 * z_log_squared_scale
        return tf.math.reduce_mean(summand, [1], name="kl_loss")


def tc_penalty(args, z_sampled, z_mean, z_log_squared_scale, prior="normal"):
    """
    From:
    Locatello, F. et al.
    Challenging Common Assumptions in the Unsupervised Learning
    of Disentangled Representations. (2018).

    Based on Equation 4 with alpha = gamma = 1 of "Isolating Sources of
    Disentanglement in Variational Autoencoders"
    (https://arxiv.org/pdf/1802.04942).
    If alpha = gamma = 1, Eq. 4 can be written as ELBO + (1 - beta) * TC.
    --

    :param args: Shared arguments
    :param z_sampled: Samples from latent space
    :param z_mean: Means of z
    :param z_log_squared_scale: Logvars of z
    :return: Total correlation penalty
    """
    tc = total_correlation(z_sampled, z_mean, z_log_squared_scale, prior)

    return (args.annealed_beta - 1.) * tc, tc


def ae_reconstruction_loss(x, logits, true_images=True):
    """Computes the Bernoulli loss."""
    flattened_dim = np.prod(x.get_shape().as_list()[1:])
    logits = tf.reshape(logits, shape=[-1, flattened_dim])
    x = tf.reshape(x, shape=[-1, flattened_dim])

    # Because true images are not binary, the lower bound in x is not zero:
    # the lower bound in x is the entropy of the true images.

    if true_images:
        dist = tfp.distributions.Bernoulli(
            probs=tf.clip_by_value(x, 1e-6, 1 - 1e-6))
        loss_lower_bound = tf.math.reduce_sum(dist.entropy(), axis=1)
    else:
        loss_lower_bound = 0

    loss = tf.math.reduce_sum(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=x),
        axis=1)

    return loss - loss_lower_bound


def gaussian_log_density(samples, mean, log_squared_scale):
    pi = tf.constant(np.pi)
    normalization = tf.math.log(2. * pi)
    inv_sigma = tf.math.exp(-log_squared_scale)
    tmp = (samples - mean)
    return -0.5 * (tmp * tmp * inv_sigma + log_squared_scale + normalization)


def laplace_log_density(samples, mean, log_squared_scale):
    c = tf.math.log(0.5)
    tmp = tf.math.abs(samples - mean)
    return c - 0.5 * log_squared_scale - tf.math.exp(-0.5 * log_squared_scale) * tmp


def total_correlation(z, z_mean, z_log_squared_scale, prior):
    """Estimate of total correlation on a batch.
    We need to compute the expectation over a batch of: E_j [log(q(z(x_j))) -
    log(prod_l q(z(x_j)_l))]. We ignore the constants as they do not matter
    for the minimization. The constant should be equal to (num_latents - 1) *
    log(batch_size * dataset_size)
    Args:
      z: [batch_size, num_latents]-tensor with sampled representation.
      z_mean: [batch_size, num_latents]-tensor with mean of the encoder.
      z_log_squared_scale: [batch_size, num_latents]-tensor with log variance of the encoder.
    Returns:
      Total correlation estimated on a batch.
    """
    # Compute log(q(z(x_j)|x_i)) for every sample in the batch, which is a
    # tensor of size [batch_size, batch_size, num_latents]. In the following
    # comments, [batch_size, batch_size, num_latents] are indexed by [j, i, l].
    if prior.lower() == "normal":
        log_qz_prob = gaussian_log_density(
            tf.expand_dims(z, 1), tf.expand_dims(z_mean, 0),
            tf.expand_dims(z_log_squared_scale, 0))
    if prior.lower() == "laplace":
        log_qz_prob = laplace_log_density(
            tf.expand_dims(z, 1), tf.expand_dims(z_mean, 0),
            tf.expand_dims(z_log_squared_scale, 0))
    # Compute log prod_l p(z(x_j)_l) = sum_l(log(sum_i(q(z(z_j)_l|x_i)))
    # + constant) for each sample in the batch, which is a vector of size
    # [batch_size,].
    log_qz_product = tf.math.reduce_sum(
        tf.math.reduce_logsumexp(log_qz_prob, axis=1, keepdims=False),
        axis=1,
        keepdims=False)
    # Compute log(q(z(x_j))) as log(sum_i(q(z(x_j)|x_i))) + constant =
    # log(sum_i(prod_l q(z(x_j)_l|x_i))) + constant.
    log_qz = tf.math.reduce_logsumexp(
        tf.math.reduce_sum(log_qz_prob, axis=2, keepdims=False),
        axis=1,
        keepdims=False)
    return tf.math.reduce_mean(log_qz - log_qz_product)
