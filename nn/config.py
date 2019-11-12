"""
parsing and configuration
"""
import os
import argparse
from nn.util import str2bool, check_folder
from time import gmtime, strftime

ROOT = os.path.abspath(os.sep)
# ROOT_ = os.path.join(ROOT, "mnt", "storage", "betaTCVAE")
ROOT_ = os.path.join(ROOT, "mnt", "mbi_data_2", "betaTCVAE")
BASE = os.path.join(ROOT_, strftime("%Y_%m_%d_%H_%M_%S", gmtime()))

def get_args(chk_dir=True, base_dir_fix=False):
    desc = "Tensorflow implementation of betaTCVAE"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument("--model_name", type=str, default="betaTCVAE",
                        help="The model name.")

    parser.add_argument("--dataset", type=str, default="helium",
                        help="Either 'helium', 'dynamic_helium','cifar10', 'fashion_mnist' or 'mnist'")

    parser.add_argument("--img_size", type=int, default=192,
                        help="The size of images")

    parser.add_argument("--img_channels", type=int, default=1,
                        help="The size of images")

    parser.add_argument("--epochs", type=int, default=int(101),
                        help="The number of epochs to run")

    parser.add_argument("--sample_size", type=int, default=int(2000),
                        help="How many sample points per epoch")

    parser.add_argument("--eval_epochs", type=int, default=1,
                        help="Eval every 'n' epoch")

    parser.add_argument("--z_dim", type=int, default=4,
                        help="Dimensionality of the latent space")

    parser.add_argument("--prior", type=str, default="normal",
                        help="Which prior to use.", choices=["normal", "laplace"])

    parser.add_argument("--batch_size", type=int, default=int(128),
                        help="The size of batch per gpu. If '-1' then we select it automagically")

    parser.add_argument("--log_steps", type=int, default=int(-1),
                        help="Log every n steps. If '-1' then we select it automagically")

    parser.add_argument("--num_warmup_steps", type=int, default=10000,
                        help="How many warmup steps should the model do")

    parser.add_argument("--lr_to_zero", type=str2bool, default=False,
                        help="Should the learning rate linearly go to zero after warmup")

    parser.add_argument("--lr_vae", type=float, default=1e-4,
                        help="learning rate")

    parser.add_argument("--beta1", type=float, default=0.9,
                        help="beta1 for Adam optimizer")

    parser.add_argument("--beta2", type=float, default=0.999,
                        help="beta2 for Adam optimizer")

    parser.add_argument("--epsilon", type=float, default=1e-8,
                        help="epsilon for the Adam optimizer")

    parser.add_argument("--beta", type=float, default=5,
                        help="beta for the TC loss")

    parser.add_argument("--weight_decay", type=float, default=.01,
                        help="l2 regularizer for training weights")

    parser.add_argument("--true_images", type=str2bool, default=True,
                        help="If True the input is not getting binarized.")

    parser.add_argument("--save_prediction_images_for_all_data", type=str2bool, default=True,
                        help="If True the predicted images are getting saved as png.")

    parser.add_argument("--do_cluster_during_eval", type=str2bool, default=True,
                        help="If False the eval routine just samples the images and saves the latent space as npy")

    parser.add_argument("--use_central_cutoff", type=str2bool, default=True,
                        help="Use only the central part of the diffraction images. "
                             "Only used when dataset == 'helium'")

    parser.add_argument("--use_self_attention", type=str2bool, default=False,
                        help="Use scalar dot-product self-attention")

    parser.add_argument("--use_residual_blocks", type=str2bool, default=True,
                        help="Use residual blocks instead of plain convolutions")

    parser.add_argument("--use_specific_weight_file", type=str, default="",
                        help="If given this file is used to load weights from.")

    parser.add_argument("--base_dir", type=str, default=BASE,
                        help="Directory name where everything will be put")

    parser.add_argument("--data_dir", type=str, default=os.path.join(ROOT_, "data"),
                        help="Where to find the data")

    return check_args(parser.parse_args(), chk_dir, base_dir_fix)


def check_args(args, chk_dir, base_dir_fix=False):
    """
    Checking arguments
    """

    # BASE
    if not base_dir_fix:
        args.base_dir += "_{}_{}_{}dims_{}".format(args.model_name, args.dataset, args.z_dim, args.prior)

    if args.use_residual_blocks:
        args.base_dir += "_using_residual_blocks"
    if args.use_self_attention:
        args.base_dir += "_with_attention"
    if chk_dir:
        check_folder(args.base_dir)

    # Create subdirectories
    args.result_dir = os.path.join(args.base_dir, "results")
    args.log_dir = os.path.join(args.base_dir, "logs")
    args.sample_dir = os.path.join(args.base_dir, "samples")

    # --result_dir
    if chk_dir:
        check_folder(args.result_dir)

    # --result_dir
    if chk_dir:
        check_folder(args.log_dir)

    # --sample_dir
    if chk_dir:
        check_folder(args.sample_dir)

    # --epoch
    try:
        assert args.epochs >= 1
    except AssertionError:
        print("\t[!] Number of epochs must be larger than or equal to one", flush=True)

    return args
