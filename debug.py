import argparse
import sys

import tensorflow as tf
from models.gan import DefenseGANBase
from utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', required=True, help='Config file')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args, _ = parser.parse_known_args()
    return args


def main(cfg, *args):
    FLAGS = tf.app.flags.FLAGS
    gan = DefenseGANBase(cfg=cfg, test_mode=True)
    gan.save_var_prefixes = ['Generator', 'Encoder']
    gan.load_generator(ckpt_path=gan.encoder_checkpoint_dir)
    gan.sess.run(gan.global_step.initializer)
    images_pl = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    rec_images = gan.reconstruct(images_pl)
    grad = tf.gradients(tf.reduce_mean(rec_images), images_pl)
    print(grad)


if __name__ == '__main__':
    args = parse_args()

    # Note: The load_config() call will convert all the parameters that are defined in
    # experiments/config files into FLAGS.param_name and can be passed in from command line.
    # arguments : python train.py --cfg <config_path> --<param_name> <param_value>
    cfg = load_config(args.cfg)
    flags = tf.app.flags

    flags.DEFINE_boolean("is_train", False,
                         "True for training, False for testing. [False]")
    flags.DEFINE_boolean("save_recs", False,
                         "True for saving reconstructions. [False]")
    flags.DEFINE_boolean("debug", False,
                         "True for debug. [False]")
    flags.DEFINE_boolean("test_generator", False,
                         "True for generator samples. [False]")
    flags.DEFINE_boolean("test_decoder", False,
                         "True for decoder samples. [False]")
    flags.DEFINE_boolean("test_again", False,
                         "True for not using cache. [False]")
    flags.DEFINE_boolean("test_batch", False,
                         "True for visualizing the batches and labels. [False]")
    flags.DEFINE_boolean("save_ds", False,
                         "True for saving the dataset in a pickle file. ["
                         "False]")
    flags.DEFINE_boolean("tensorboard_log", True, "True for saving "
                                                  "tensorboard logs. [True]")
    flags.DEFINE_boolean("train_encoder", False,
                         "Add an encoder to a pretrained model. ["
                         "False]")
    flags.DEFINE_boolean("test_encoder", False, "Test encoder. [False]")
    flags.DEFINE_boolean("init_with_enc", False,
                         "Initializes the z with an encoder, must run "
                         "--train_encoder first. [False]")
    flags.DEFINE_integer("max_num", -1,
                         "True for saving the dataset in a pickle file ["
                         "False]")
    flags.DEFINE_string("init_path", None, "Checkpoint path. [None]")

    main_cfg = lambda x: main(cfg, x)
    tf.app.run(main=main_cfg)