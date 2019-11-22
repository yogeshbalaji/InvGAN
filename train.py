# Copyright 2018 The Defense-GAN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""The main class for training GANs."""

import argparse
import sys

import tensorflow as tf

from utils.config import load_config, gan_from_config
from utils.reconstruction import reconstruct_dataset, save_ds, \
    encoder_reconstruct, evaluate_encoder
from utils.metrics import compute_inception_score, save_mse


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
    test_mode = not (FLAGS.is_train or FLAGS.train_encoder)
    gan = gan_from_config(cfg, test_mode)

    if FLAGS.is_train:
        gan.train()

    if FLAGS.save_recs:
        gan.load_model()
        ret_all = reconstruct_dataset(gan_model=gan, ckpt_path=FLAGS.init_path,
                                      max_num=FLAGS.max_num)
        save_mse(reconstruction_dict=ret_all, gan_model=gan)

    if FLAGS.test_generator:
        compute_inception_score(gan_model=gan, ckpt_path=FLAGS.init_path)

    if FLAGS.eval_encoder:
        gan.load_model()
        evaluate_encoder(gan, FLAGS.output_name)

    if FLAGS.test_encoder:
        gan.load_model()
        (train_error, dev_error, test_error) = encoder_reconstruct(
            gan_model=gan)

        ## Logging the error
        logfile = open('output/encoder_results.txt', 'a+')
        config_ = 'loss_{}_lr_{}\n'.format(gan.encoder_loss_type,
                                           gan.encoder_lr)
        losses_ = 'Train loss: {}, Dev loss: {}, Test loss: {}\n\n\n'.format(
            train_error, dev_error, test_error)
        logfile.writelines(config_)
        logfile.writelines(losses_)
        logfile.close()

    if FLAGS.test_batch:
        gan.test_batch()

    if FLAGS.save_ds:
        save_ds(gan_model=gan)

    gan.close_session()


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
    flags.DEFINE_boolean("test_encoder", False, "Test encoder. [False]")
    flags.DEFINE_boolean("eval_encoder", False, "Evaluate encoder. [False]")
    flags.DEFINE_boolean("train_encoder", False, "Train encoder. [False]")
    flags.DEFINE_boolean("init_with_enc", False,
                         "Initializes the z with an encoder, must run "
                         "--train_encoder first. [False]")
    flags.DEFINE_integer("max_num", -1,
                         "True for saving the dataset in a pickle file ["
                         "False]")
    flags.DEFINE_string("init_path", None, "Checkpoint path. [None]")
    flags.DEFINE_string("output_name", 'all', "Output filename for encoder evaluation.")


    main_cfg = lambda x: main(cfg, x)
    tf.app.run(main=main_cfg)
