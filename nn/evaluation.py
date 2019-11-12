import os

import warnings
import numpy as np
import tensorflow as tf
from itertools import combinations, product
from scipy.stats import norm, laplace
from matplotlib.pyplot import close
from sklearn import metrics
from sklearn import mixture
from sklearn.preprocessing import StandardScaler
from ugtm import eGTM

from nn.data import get_data
from nn.visualize import plot_cluster, image_grid, clean_img, bar_plot
from nn.util import check_folder


def eval_model(args, betaTCVAE, epoch, dir_str=None):
    step = args.global_step
    if dir_str is None:
        dir_str = "epoch_{}".format(epoch)
    step_folder = os.path.join(args.sample_dir, dir_str)
    check_folder(step_folder)

    if step == 0:
        return

    args.logger.info("Sampling the model")
    images, z, lbl = sample_model(args, betaTCVAE)
    args.logger.info("...done")

    args.logger.info("Checking folder")
    npy_path = os.path.join(step_folder, "npy")
    emb_path = os.path.join(step_folder, "emb")
    check_folder(npy_path)
    check_folder(emb_path)
    args.logger.info("..done")

    args.logger.info("Getting label information")
    if args.dataset.lower() == "helium":
        lbl, rad = (lbl[0], lbl[1])

    if args.dataset.lower() == "dynamic_helium":
        there_is_truth = False  # There is no truth when it comes to dynamic helium
    else:
        there_is_truth = True
    args.logger.info("..done")

    true_img = np.array([x.numpy() for x in images[0]])
    gen_img = np.array([x.numpy() for x in images[1]])
    if args.do_cluster_during_eval:
        args.logger.info("Clustering the data")
        warnings.filterwarnings("ignore")
        cluster, figs, titles, final_ttl = cluster_z(args, z, lbl, there_is_truth)
        warnings.filterwarnings("default")
        args.logger.info("...done")

        # Benchmark the clustering
        if there_is_truth:
            args.logger.info("Benchmark the clustering with ground truth labels")
            for c, t in zip(cluster, titles):
                try:
                    benchmark(args, c, lbl, t)
                except Exception as e:
                    args.logger.info(e)

        args.logger.info("Benchmark the clustering without ground truth labels")
        for c, t in zip(cluster, titles):
            try:
                unsupervised_benchmark(args, z, c, t)
            except Exception as e:
                args.logger.info(e)

        # Plot the embeddings
        for ttl, fig in zip(final_ttl, figs):
            path_ = os.path.join(emb_path, "{}_{}".format(step, ttl))
            fig.savefig(path_)
            close(fig)

        # Take the first clustering label array - that always belongs to a GMM
        # clustering. An write out the images corresponding to their label
        args.logger.info("Writeout out the clustering")
        gmm_lbl = cluster[0].squeeze()
        # Get all unique label
        u_lbl, c_lbl = np.unique(gmm_lbl, return_counts=True)

        clustering_path = os.path.join(step_folder, "clustering")
        if not os.path.isdir(clustering_path):
            os.mkdir(clustering_path)
        lbl_sorted_true_images = [true_img[gmm_lbl == x] for x in u_lbl]
        lbl_sorted_gen_images = [gen_img[gmm_lbl == x] for x in u_lbl]

        lbl_sorted_true_label = [lbl[gmm_lbl == x] for x in u_lbl]
        if there_is_truth:
            n, c = np.unique(lbl, return_counts=True)
            if args.dataset.lower() == "helium":
                class_names = ["Round", "Elliptical", "Bent", "Asymmetric", "Double Rings", "Streak"]
            else:
                class_names = ["{}".format(x) for x in range(n.shape[0])]
            class_map = {}
            for n_, c_ in zip(n, c):
                class_map.update({str(n_): c_})

        for u_, true_, gen_, true_lbl in zip(u_lbl, lbl_sorted_true_images,
                                             lbl_sorted_gen_images,
                                             lbl_sorted_true_label):
            out_img_path_ = os.path.join(clustering_path, "{}".format(u_))
            if not os.path.isdir(out_img_path_):
                os.mkdir(out_img_path_)

            if there_is_truth:
                cluster_class_n, cluster_class_c = np.unique(true_lbl, return_counts=True)

                names = [class_names[x] for x in cluster_class_n]
                class_counts = [class_map[str(x)] for x in cluster_class_n]

                x_bars = (cluster_class_n - .11, cluster_class_n + .11)
                y_bars = (cluster_class_c / class_counts, cluster_class_c / true_lbl.shape[0])
                labels = ("Percentage of original class that are in this GMM class",
                          "Percentage how much this original class contributes to this GMM class")
                ticks = cluster_class_n

                f = bar_plot(x_bars, y_bars, labels, ticks, names)
                f.savefig(os.path.join(clustering_path, "{}".format(u_)))
                close(f)

            if u_ != -1 and args.save_prediction_images_for_all_data:
                for ii, (true_single, gen_single, true_lbl_) in enumerate(zip(true_, gen_, true_lbl)):
                    f = clean_img((true_single, gen_single))
                    f.savefig(os.path.join(out_img_path_, "{}_{}".format(true_lbl_, ii)))
                    close(f)

    args.logger.info("Saving the sample images and the reconstructions")
    titles = ("x", "xhat")
    grid_size = 25

    grid_export = [true_img[:100], gen_img[:100]]
    # Export x and xhat
    for img, title in zip(grid_export, titles):
        for ii, chunk in enumerate(chunks(img, grid_size)):
            if len(chunk) == grid_size:
                f = image_grid(chunk, grid_size=grid_size)
                f.savefig(os.path.join(step_folder, "{}_{}".format(ii, title)))
                close(f)

    if args.z_dim == 2:
        f = image_grid(images[-1], grid_size=625)
        f.savefig(os.path.join(step_folder, "interp_z"))
        close(f)

    if 2 < args.z_dim <= 6:
        idxs = list(combinations(np.arange(args.z_dim, dtype=int), 2))
        fix_idxs = [0.05, 0.25, 0.5, 0.75, 0.95]
        for img_scan, (zd, fd) in zip(images[-1], product(idxs, fix_idxs)):
            ttl_ = "Scanned across z-dim: {} with other dims fixed at: {}".format(zd, fd)
            f = image_grid(img_scan, grid_size=625, ttl=ttl_)
            zd_ = str(zd).replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
            fd_ = str(fd).replace(".", "")
            f.savefig(os.path.join(step_folder, "interp_z_{}_{}".format(zd_, fd_)))
            close(f)

    args.logger.info("Saving the embeddings to npy file")
    path = os.path.join(npy_path, "{}_z".format(step))

    if args.dataset.lower() == "helium":
        np.savez(path, z=z, lbl=lbl, true_imgs=true_img, gen_imgs=gen_img, radius=rad)
    else:
        np.savez(path, z=z, lbl=lbl, true_imgs=true_img, gen_imgs=gen_img)


def get_scores(lbl):
    hs = metrics.homogeneity_score(lbl[0], lbl[1]),
    cs = metrics.completeness_score(lbl[0], lbl[1]),
    rs = metrics.adjusted_rand_score(lbl[0], lbl[1]),
    mis = metrics.normalized_mutual_info_score(lbl[0], lbl[1], average_method="arithmetic"),

    names = ["homogeneity_score", "completeness_score",
             "adjusted_rand_score", "normalized_mutual_info_score"]
    metrics_ = [hs, cs, rs, mis]

    return metrics_, names,


def benchmark(args, estimated_labels, true_labels, estimator_name):
    # Separate out non-labelled data points
    non_zero_estimated_label_idx = (estimated_labels >= 0)
    non_zero_estimated_labels = estimated_labels[non_zero_estimated_label_idx]
    non_zero_true_labels = true_labels[non_zero_estimated_label_idx]

    # Iterate over a tuple with two (true, pred) pairs
    it_ = ((true_labels, estimated_labels), (non_zero_true_labels, non_zero_estimated_labels))
    ttl_ = ("full", "non_zero")

    for ttl, lbl in zip(ttl_, it_):
        metrics_, names = get_scores(lbl)

        for l, n in zip(metrics_, names):
            s_ = "{}/{}/{}".format(ttl, estimator_name, n)
            tf.summary.scalar(s_, np.squeeze(l), step=args.global_step)


def unsupervised_benchmark(args, data, estimated_labels, estimator_name):
    # Separate out non-labelled data points

    db_score = metrics.davies_bouldin_score(data, estimated_labels)
    ch_score = metrics.calinski_harabasz_score(data, estimated_labels)
    sil_score = metrics.silhouette_score(data, estimated_labels)

    for l, n in zip([db_score, ch_score, sil_score], ["db_score", "ch_score", "sil_score"]):
        s_ = "unsupervised/{}/{}".format(estimator_name, n)
        tf.summary.scalar(s_, np.squeeze(l), step=args.global_step)


def sample_model(args, betaTCVAE):
    """
    Sample the model
    """

    test_x = get_data(args, "test")
    x = []
    xhat = []
    z = []
    lbl = []
    if args.dataset.lower() == "helium":
        rad = []
    for ii, x_test in enumerate(test_x):
        x.extend(x_test[0])

        logits, _, (z_mean, log_var) = betaTCVAE.model(x_test)
        xhat.extend(tf.math.sigmoid(logits))
        # we "comb" the mean and the std of every dist in z together
        # so that z looks like: [z1_mean, z1_std, z2_mean, z2_std, ...]
        z_std = tf.math.exp(log_var * .5)

        # print(np.shape(out[0]), np.shape(out[1]), np.shape(z_mean), np.shape(z_std))
        z_combed = np.zeros([np.shape(z_std)[0], int(args.z_dim * 2)])
        # idx_even = np.arange(0, int(args.z_dim * 2), 2)
        # print(np.shape(z_combed), np.shape(idx_even))

        z_combed[:, ::2] = z_mean
        z_combed[:, 1::2] = z_std

        z.extend(z_combed)

        # we concatenate mean and std
        # z.extend(np.hstack([z_mean, z_std]))
        if args.dataset.lower() == "helium":
            lbl.extend(x_test[-2])
            rad.extend(x_test[-1])
        else:
            lbl.extend(x_test[-1])

    z = np.array(z).reshape(2000, int(args.z_dim * 2))
    if args.dataset.lower() == "helium":
        lbl = np.array(lbl).reshape([-1])
        rad = np.array(rad).reshape([-1])
        lbl = (lbl, rad)
    else:
        lbl = np.array(lbl).reshape([-1])

    if args.z_dim <= 6:
        if args.prior.lower() == "normal":
            grid_x = norm.ppf(np.linspace(0.05, 0.95, 25))
        if args.prior.lower() == "laplace":
            grid_x = laplace.ppf(np.linspace(0.05, 0.95, 25))
        z_ppf = np.vstack([pp.reshape(-1) for pp in np.meshgrid(grid_x, grid_x)]).T

    if args.z_dim == 2:
        x_decoded = betaTCVAE.decode(z_ppf, apply_sigmoid=True)
        return (x, xhat, x_decoded), z, lbl

    if 2 < args.z_dim <= 6:
        # We make grids for all 2-d permutations, while keeping the other dims fixed.
        # Fixations are ppl(0.05), ppl(0.25), ppl(0.5), ppl(0.75) and ppl(0.95).
        # This means we end up with (z-dim over 2) * 5 grid maps -> with z_dim = 6, we get 75 images
        x_decoded = []
        if args.prior.lower() == "normal":
            fixed_z = norm.ppf([0.05, 0.25, 0.5, 0.75, 0.95])
        if args.prior.lower() == "laplace":
            fixed_z = laplace.ppf([0.05, 0.25, 0.5, 0.75, 0.95])

        for comb in combinations(np.arange(args.z_dim, dtype=int), 2):
            # args.logger.info(comb)
            for fixed_latent in fixed_z:
                sample_z = np.ones([625, args.z_dim]) * fixed_latent
                sample_z[:, int(comb[0])] = z_ppf[:, 0]
                sample_z[:, int(comb[1])] = z_ppf[:, 1]
                x_decoded.append(betaTCVAE.decode(sample_z, apply_sigmoid=True))
        # x_decoded = np.array(x_decoded)
        return (x, xhat, x_decoded), z, lbl

    return (x, xhat), z, lbl


def cluster_z(args, z, true_lbl, plot_truth=True):
    cluster = []
    titles = []
    figs = []
    final_ttl = []

    scaler = StandardScaler(copy=True, with_mean=True, with_std=False)
    scaled_z = scaler.fit_transform(z)
    if args.z_dim != 2:
        z_map = eGTM(k=16, m=4,
                     s=0.3, regul=2.,
                     random_state=1612,
                     niter=400).fit_transform(scaled_z, model='means')
    else:
        z_map = z

    args.logger.info("Clustering z directly with a Gaussian Mixture Model")
    gmm_z = mixture.GaussianMixture(n_components=6, covariance_type='full').fit_predict(scaled_z)
    cluster.append(gmm_z)
    titles.append("Clustering z directly with a Gaussian Mixture Model")

    non_zero_estimated_label_idx = (gmm_z >= 0)
    non_zero_estimated_labels = gmm_z[non_zero_estimated_label_idx]
    non_zero_visualization = z_map[non_zero_estimated_label_idx]

    try:
        class_names = ["{}".format(x) for x in np.unique(non_zero_estimated_labels)]
        figs.append(plot_cluster(non_zero_visualization, non_zero_estimated_labels,
                                 "Gaussian mixture model on latent space (Non-zero classes)",
                                 legend_labels=class_names))
        final_ttl.append("gmm_z_non_zero")
    except Exception as e:
        # args.logger.info(e)  # log, but keep going on
        # args.logger.info("found only {} classes\n".format(c.max()))
        pass

    # args.logger.info("\t all ")
    try:
        class_names = ["{}".format(x) for x in np.unique(gmm_z)]
        figs.append(plot_cluster(z_map, gmm_z, "Gaussian mixture model on latent space (All classes)",
                                 legend_labels=class_names))
        final_ttl.append("gmm_z_all")
    except Exception as e:
        # args.logger.info(e)  # log, but keep going on
        # args.logger.info("found only {} classes\n".format(c.max()))
        pass

    if plot_truth:
        if args.dataset.lower() == "helium":
            class_names = ["Round", "Elliptical", "Bent", "Asymmetric", "Double Rings", "Streak"]
        else:
            class_names = ["{}".format(x) for x in np.unique(true_lbl)]
        figs.append(plot_cluster(z_map, true_lbl, "Generative topographic map of the VAEs latent space", s=150,
                                 legend_labels=class_names))
        final_ttl.append("gtm_truth")

    try:
        assert np.shape(cluster)[0] == np.shape(titles)[0]
    except AssertionError as e:
        args.logger.info(e)
        args.logger.info((np.shape(cluster), np.shape(titles)))

    try:
        assert np.shape(final_ttl)[0] == np.shape(figs)[0]
    except AssertionError as e:
        args.logger.info(e)
        args.logger.info((np.shape(final_ttl), np.shape(figs)))

    return cluster, figs, titles, final_ttl


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
