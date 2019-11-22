import argparse
import cPickle
import sys
import os

import tensorflow as tf
from utils.config import load_config, gan_from_config
from classifiers.cifar_model import Model
from blackbox import get_cached_gan_data
from utils.network_builder import model_a, DefenseWrapper
from cleverhans.utils_tf import model_train, model_eval
from utils.gan_defense import model_eval_gan
from utils.misc import ensure_dir
import numpy as np


BATCH_SIZE = 128


def main(cfg, *args):
    FLAGS = tf.app.flags.FLAGS

    rng = np.random.RandomState([11, 24, 1990])
    tf.set_random_seed(11241990)

    gan = gan_from_config(cfg, True)

    results_dir = 'results/clean/{}'.format(gan.dataset_name)
    ensure_dir(results_dir)

    sess = gan.sess
    gan.load_model()

    # use test split
    train_images, train_labels, test_images, test_labels = get_cached_gan_data(gan, test_on_dev=False, orig_data_flag=True)

    x_shape = [None] + list(train_images.shape[1:])
    images_pl = tf.placeholder(tf.float32, shape=[BATCH_SIZE] + list(train_images.shape[1:]))
    labels_pl = tf.placeholder(tf.float32, shape=[BATCH_SIZE] + [train_labels.shape[1]])

    if FLAGS.num_tests > 0:
        test_images = test_images[:FLAGS.num_tests]
        test_labels = test_labels[:FLAGS.num_tests]

    if FLAGS.num_train > 0:
        train_images = train_images[:FLAGS.num_train]
        train_labels = train_labels[:FLAGS.num_train]

    train_params = {
        'nb_epochs': 10,
        'batch_size': BATCH_SIZE,
        'learning_rate': 0.001}

    eval_params = {'batch_size': BATCH_SIZE}

    # train classifier for mnist, fmnist
    if gan.dataset_name in ['mnist', 'f-mnist']:
        model = model_a(input_shape=x_shape, nb_classes=train_labels.shape[1])
        preds_train = model.get_logits(images_pl, dropout=True)

        model_train(sess, images_pl, labels_pl, preds_train, train_images, train_labels,
                    args=train_params, rng=rng, init_all=False)

    elif gan.dataset_name == 'cifar-10':
        pre_model = Model('classifiers/model/', tiny=False, mode='eval', sess=sess)
        model = DefenseWrapper(pre_model, 'logits')

    elif gan.dataset_name == 'celeba':
        # TODO
        raise NotImplementedError

    model.add_rec_model(gan, batch_size=BATCH_SIZE)
    preds_eval = model.get_logits(images_pl)

    # calculate norms
    num_dims = len(images_pl.get_shape())
    avg_inds = list(range(1, num_dims))
    reconstruct = gan.reconstruct(images_pl, batch_size=BATCH_SIZE)

    # We use L2 loss for GD steps
    diff_op = tf.reduce_mean(tf.square(reconstruct - images_pl), axis=avg_inds)

    acc, mse, roc_info = model_eval_gan(sess, images_pl, labels_pl, preds_eval, None, test_images=test_images,
                                  test_labels=test_labels, args=eval_params, diff_op=diff_op)
    # Logging
    logfile = open(os.path.join(results_dir, 'acc.txt'), 'a+')
    msg = 'lr_{}_iters_{}, {}\n'.format(gan.rec_lr, gan.rec_iters, acc)
    logfile.writelines(msg)
    logfile.close()

    logfile = open(os.path.join(results_dir, 'mse.txt'), 'a+')
    msg = 'lr_{}_iters_{}, {}\n'.format(gan.rec_lr, gan.rec_iters, mse)
    logfile.writelines(msg)
    logfile.close()

    pickle_filename = os.path.join(results_dir, 'roc_lr_{}_iters_{}.pkl'.format(gan.rec_lr, gan.rec_iters))
    with open(pickle_filename, 'w') as f:
        cPickle.dump(roc_info, f, cPickle.HIGHEST_PROTOCOL)
        print('[*] saved roc_info in {}'.format(pickle_filename))

    return [acc, mse]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', required=True, help='Config file')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args, _ = parser.parse_known_args()
    return args


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
    flags.DEFINE_integer("max_num", -1,
                         "True for saving the dataset in a pickle file ["
                         "False]")
    flags.DEFINE_integer("num_train", -1, 'Number of training data to load.')
    flags.DEFINE_integer('num_tests', -1, 'Number of test samples.')
    flags.DEFINE_string("init_path", None, "Checkpoint path. [None]")

    main_cfg = lambda x: main(cfg, x)
    tf.app.run(main=main_cfg)