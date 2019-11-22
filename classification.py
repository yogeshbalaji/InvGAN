import argparse
import sys
import os

import tensorflow as tf
from utils.config import load_config, gan_from_config
from classifiers.cifar_model import Model
from blackbox import get_cached_gan_data
from utils.network_builder import model_a, DefenseWrapper
from cleverhans.utils_tf import model_train, model_eval
from utils.misc import ensure_dir
import numpy as np


def main(cfg, *args):
    FLAGS = tf.app.flags.FLAGS

    rng = np.random.RandomState([11, 24, 1990])

    gan = gan_from_config(cfg, True)

    results_dir = 'results/sweep/{}'.format(gan.dataset_name)
    ensure_dir(results_dir)

    sess = gan.sess
    gan.load_model()

    # Evaluate on dev set
    train_images, train_labels, test_images, test_labels = get_cached_gan_data(gan, test_on_dev=True, orig_data_flag=True)

    x_shape = [None] + list(train_images.shape[1:])
    images_pl = tf.placeholder(tf.float32, shape=[None] + list(train_images.shape[1:]))
    labels_pl = tf.placeholder(tf.float32, shape=[None] + [train_labels.shape[1]])

    train_params = {
        'nb_epochs': 10,
        'batch_size': 128,
        'learning_rate': 0.001}

    eval_params = {'batch_size': 128}

    # train classifier for mnist, fmnist
    if gan.dataset_name in ['mnist', 'f-mnist']:
        model = model_a(input_shape=x_shape, nb_classes=train_labels.shape[1])
        preds_train = model.get_logits(images_pl, dropout=True)
        preds_eval = model.get_logits(images_pl)

        tf.set_random_seed(11241990)

        model_train(sess, images_pl, labels_pl, preds_train, train_images, train_labels,
                    args=train_params, rng=rng, init_all=False)

    elif gan.dataset_name == 'cifar-10':
        pre_model = Model('classifiers/model/', tiny=False, mode='eval', sess=sess)
        model = DefenseWrapper(pre_model, 'logits')

        preds_eval = model.get_logits(images_pl)

    train_acc = model_eval(sess, images_pl, labels_pl, preds_eval, train_images, train_labels,
                               args=eval_params)
    eval_acc = model_eval(sess, images_pl, labels_pl, preds_eval, test_images, test_labels,
                              args=eval_params)

    model.add_rec_model(gan, batch_size=128)
    preds_eval = model.get_logits(images_pl)
    tf.set_random_seed(11241990)
    sess.run(tf.local_variables_initializer())

    eval_rec_acc = model_eval(sess, images_pl, labels_pl, preds_eval, test_images, test_labels,
                              args=eval_params)
    # Logging
    logfile = open(os.path.join(results_dir, 'acc_train.txt'), 'a+')
    msg = 'iters_{}_lr_{}, {:6f}\n'.format(gan.rec_iters, gan.rec_lr, train_acc)
    logfile.writelines(msg)
    logfile.close()

    logfile = open(os.path.join(results_dir, 'acc_eval.txt'), 'a+')
    msg = 'iters_{}_lr_{}, {:6f}\n'.format(gan.rec_iters, gan.rec_lr, eval_acc)
    logfile.writelines(msg)
    logfile.close()

    logfile = open(os.path.join(results_dir, 'acc_eval_rec.txt'), 'a+')
    msg = 'iters_{}_lr_{}, {:6f}\n'.format(gan.rec_iters, gan.rec_lr, eval_rec_acc)
    logfile.writelines(msg)
    logfile.close()

    return [train_acc, eval_acc, eval_rec_acc]


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
    flags.DEFINE_string("init_path", None, "Checkpoint path. [None]")

    main_cfg = lambda x: main(cfg, x)
    tf.app.run(main=main_cfg)