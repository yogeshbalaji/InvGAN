# Copyright 2019 The Inv-GAN Authors. All Rights Reserved.
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
# ==============================================================================

"""Testing white-box attacks Inv-GAN models. This module is based on MNIST
tutorial of cleverhans."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import _init_paths

import argparse
import cPickle
import logging
import os
import sys

import numpy as np
import tensorflow as tf

from classifiers.cifar_model import Model
from blackbox import get_cached_gan_data, get_reconstructor
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks.bpda import BPDAL2
from utils.attack import MadryEtAl
from cleverhans.utils import set_log_level, AccuracyReport
from cleverhans.utils_tf import model_train, model_eval
from models.gan_v2 import InvertorDefenseGAN
from models.dataset_networks import get_generator_fn
from utils.config import load_config, gan_from_config
from utils.gan_defense import model_eval_gan
from utils.misc import ensure_dir
from utils.network_builder import model_a, model_b, model_c, model_d, model_e, model_f, model_y, DefenseWrapper

orig_data_paths = {k: 'data/cache/{}_pkl'.format(k) for k in ['mnist', 'f-mnist', 'cifar-10']}
attack_config_dict = {'mnist': {'eps': 0.3, 'clip_min': 0},
                      'f-mnist': {'eps': 0.3, 'clip_min': 0},
                      'cifar-10': {'eps': 8*2 / 255.0, 'clip_min': -1},
                      'celeba': {'eps': 8*2 / 255.0, 'clip_min': -1}
                      }


def get_diff_op(classifier, x1, x2, use_image=False):

    if use_image:
        f1 = x1
        f2 = x2
    else:
        f1 = classifier.extract_feature(x1)
        f2 = classifier.extract_feature(x2)

    num_dims = len(f1.get_shape())
    avg_inds = list(range(1, num_dims))

    return tf.reduce_mean(tf.square(f1 - f2), axis=avg_inds)


def whitebox(gan, rec_data_path=None, batch_size=128, learning_rate=0.001,
             nb_epochs=10, eps=0.3, online_training=False,
             test_on_dev=False, attack_type='fgsm', defense_type='gan',
             num_tests=-1, num_train=-1, cfg=None):
    """Based on MNIST tutorial from cleverhans.
    
    Args:
         gan: A `GAN` model.
         rec_data_path: A string to the directory.
         batch_size: The size of the batch.
         learning_rate: The learning rate for training the target models.
         nb_epochs: Number of epochs for training the target model.
         eps: The epsilon of FGSM.
         online_training: Training Defense-GAN with online reconstruction. The
            faster but less accurate way is to reconstruct the dataset once and use
            it to train the target models with:
            `python train.py --cfg <path-to-model> --save_recs`
         attack_type: Type of the white-box attack. It can be `fgsm`,
            `rand+fgsm`, or `cw`.
         defense_type: String representing the type of attack. Can be `none`,
            `defense_gan`, or `adv_tr`.
    """
    
    FLAGS = tf.flags.FLAGS
    rng = np.random.RandomState([11, 24, 1990])

    # Set logging level to see debug information.
    set_log_level(logging.WARNING)

    ### Attack paramters
    eps = attack_config_dict[gan.dataset_name]['eps']
    min_val = attack_config_dict[gan.dataset_name]['clip_min']
    attack_iterations = FLAGS.attack_iters
    search_steps = FLAGS.search_steps

    train_images, train_labels, test_images, test_labels = get_cached_gan_data(gan, test_on_dev, orig_data_flag=True)

    sess = gan.sess
    # if defense_type == 'defense_gan':
    #     assert gan is not None
    #     sess = gan.sess
    #
    #     if FLAGS.train_on_recs:
    #         assert rec_data_path is not None or online_training
    # else:
    #     config = tf.ConfigProto()
    #     config.gpu_options.allow_growth = True
    #     sess = tf.Session(config=config)

    # Classifier is trained on either original data or reconstructed data.
    # During testing, the input image will be reconstructed by GAN.
    # Therefore, we use rec_test_images as input to the classifier.
    # When evaluating defense_gan with attack, input should be test_images.

    x_shape = [None] + list(train_images.shape[1:])
    images_pl = tf.placeholder(tf.float32, shape=[None] + list(train_images.shape[1:]))
    labels_pl = tf.placeholder(tf.float32, shape=[None] + [train_labels.shape[1]])

    if num_tests > 0:
        test_images = test_images[:num_tests]
        test_labels = test_labels[:num_tests]

    if num_train > 0:
        train_images = train_images[:num_train]
        train_labels = train_labels[:num_train]

    # Creating classificaion model

    if gan.dataset_name in ['mnist', 'f-mnist']:
        images_pl_transformed = images_pl
        models = {'A': model_a, 'B': model_b, 'C': model_c, 'D': model_d, 'E': model_e, 'F': model_f}

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            model = models[FLAGS.model](input_shape=x_shape, nb_classes=train_labels.shape[1])

        used_vars = model.get_params()
        preds_train = model.get_logits(images_pl_transformed, dropout=True)
        preds_eval = model.get_logits(images_pl_transformed)

    elif gan.dataset_name == 'cifar-10':
        images_pl_transformed = images_pl
        pre_model = Model('classifiers/model/cifar-10', tiny=False, mode='eval', sess=sess)
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            model = DefenseWrapper(pre_model, 'logits')

        used_vars = [x for x in tf.global_variables() if x.name.startswith('model')]
        preds_eval = model.get_logits(images_pl_transformed)

    elif gan.dataset_name == 'celeba':
        images_pl_transformed = tf.cast(images_pl, tf.float32) / 255. * 2. - 1.
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            model = model_y(input_shape=x_shape, nb_classes=train_labels.shape[1])

        used_vars = model.get_params()
        preds_train = model.get_logits(images_pl_transformed, dropout=True)
        preds_eval = model.get_logits(images_pl_transformed)

    # Creating BPDA model
    if attack_type in ['bpda', 'bpda-pgd']:
        gan_bpda = InvertorDefenseGAN(
            get_generator_fn(cfg['DATASET_NAME'], cfg['USE_RESBLOCK']), cfg=cfg,
            test_mode=True)
        gan_bpda.checkpoint_dir = cfg['BPDA_ENCODER_CP_PATH']
        gan_bpda.generator_init_path = cfg['BPDA_GENERATOR_INIT_PATH']
        gan_bpda.active_sess = sess
        gan_bpda.load_model()

        if gan.dataset_name in ['mnist', 'f-mnist']:
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                attack_model = models[FLAGS.model](input_shape=x_shape, nb_classes=train_labels.shape[1])
            attack_used_vars = attack_model.get_params()
        elif gan.dataset_name == 'cifar-10':
            pre_model_attack = Model('classifiers/model/cifar-10', tiny=False, mode='eval', sess=sess)
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                attack_model = DefenseWrapper(pre_model_attack, 'logits')
            attack_used_vars = [x for x in tf.global_variables() if x.name.startswith('model')]
        elif gan.dataset_name == 'celeba':
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                attack_model = model_y(input_shape=x_shape, nb_classes=train_labels.shape[1])
            attack_used_vars = attack_model.get_params()

    report = AccuracyReport()

    def evaluate():
        # Evaluate the accuracy of the MNIST model on legitimate test
        # examples.
        eval_params = {'batch_size': batch_size}
        acc = model_eval(
            sess, images_pl, labels_pl, preds_eval, test_images, test_labels, args=eval_params)
        report.clean_train_clean_eval = acc
        print('Test accuracy: %0.4f' % acc)

    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'train_dir': 'classifiers/model/{}'.format(gan.dataset_name),
        'filename': 'model_{}'.format(FLAGS.model)
    }

    preds_adv = None
    if FLAGS.defense_type == 'adv_tr':
        attack_params = {'eps': FLAGS.fgsm_eps_tr,
                         'clip_min': 0.,
                         'clip_max': 1.}
        if gan:
            if gan.dataset_name == 'celeba':
                attack_params['clip_min'] = -1.0

        attack_obj = FastGradientMethod(model, sess=sess)
        adv_x_tr = attack_obj.generate(images_pl_transformed, **attack_params)
        adv_x_tr = tf.stop_gradient(adv_x_tr)
        preds_adv = model(adv_x_tr)

    classifier_load_success = False
    if FLAGS.load_classifier:
        try:
            path = tf.train.latest_checkpoint('classifiers/model/{}'.format(gan.dataset_name))
            saver = tf.train.Saver(var_list=used_vars)
            saver.restore(sess, path)
            print('[+] Classifier loaded successfully ...')
            classifier_load_success = True
        except:
            print('[-] Cannot load classifier ...')
            classifier_load_success = False

    if classifier_load_success == False:
        print('[+] Training classifier model ...')
        model_train(sess, images_pl, labels_pl, preds_train, train_images, train_labels,
                args=train_params, rng=rng, predictions_adv=preds_adv,
                init_all=False, evaluate=evaluate, save=False)

    if attack_type in ['bpda', 'bpda-pgd']:
        # Initialize attack model weights with trained model
        path = tf.train.latest_checkpoint('classifiers/model/{}'.format(gan.dataset_name))
        saver = tf.train.Saver(var_list=attack_used_vars)
        saver.restore(sess, path)
        print('[+] Attack model initialized successfully ...')
        
        # Add self.enc_reconstruction
        # Only auto-encodes to reconstruct. No GD is performed
        attack_model.add_rec_model(gan_bpda, batch_size, ae_flag=True)
        
    # Calculate training error.
    eval_params = {'batch_size': batch_size}

    # Evaluate trained model
    #train_acc = model_eval(sess, images_pl, labels_pl, preds_eval, train_images, train_labels,
    #                       args=eval_params)
    # print('[#] Train acc: {}'.format(train_acc))

    eval_acc = model_eval(sess, images_pl, labels_pl, preds_eval, test_images, test_labels,
                          args=eval_params)
    print('[#] Eval acc: {}'.format(eval_acc))

    reconstructor = get_reconstructor(gan)

    if attack_type is None:
        return eval_acc, 0, None

    if 'rand' in FLAGS.attack_type:
        test_images = np.clip(
            test_images + args.alpha * np.sign(np.random.randn(*test_images.shape)),
            min_val, 1.0)
        eps -= args.alpha

    if 'fgsm' in FLAGS.attack_type:
        attack_params = {'eps': eps, 'ord': np.inf, 'clip_min': min_val, 'clip_max': 1.}
        attack_obj = FastGradientMethod(model, sess=sess)
    elif FLAGS.attack_type == 'cw':
        attack_obj = CarliniWagnerL2(model, sess=sess)
        attack_params = {'binary_search_steps': 6,
                         'max_iterations': attack_iterations,
                         'learning_rate': 0.2,
                         'batch_size': batch_size,
                         'clip_min': min_val,
                         'clip_max': 1.,
                         'initial_const': 10.0}

    elif FLAGS.attack_type == 'madry':
        attack_obj = MadryEtAl(model, sess=sess)
        attack_params = {'eps': eps, 'eps_iter': eps / 4.0, 'clip_min': min_val, 'clip_max': 1.,
                         'ord': np.inf, 'nb_iter': attack_iterations}
    
    elif FLAGS.attack_type == 'bpda':
        # BPDA + FGSM
        attack_params = {'eps': eps, 'ord': np.inf, 'clip_min': min_val, 'clip_max': 1.}
        attack_obj = FastGradientMethod(attack_model, sess=sess)

    elif FLAGS.attack_type == 'bpda-pgd':
        # BPDA + PGD
        attack_params = {'eps': eps, 'eps_iter': eps / 4.0, 'clip_min': min_val, 'clip_max': 1.,
                         'ord': np.inf, 'nb_iter': attack_iterations}
        attack_obj = MadryEtAl(attack_model, sess=sess)

    elif FLAGS.attack_type == 'bpda-l2':
        # default: lr=1.0, c=0.1
        attack_obj = BPDAL2(model, reconstructor, sess=sess)
        attack_params = {'binary_search_steps': search_steps,
                         'max_iterations': attack_iterations,
                         'learning_rate': 0.2,
                         'batch_size': batch_size,
                         'clip_min': min_val,
                         'clip_max': 1.,
                         'initial_const': 10.0}

    adv_x = attack_obj.generate(images_pl_transformed, **attack_params)

    if FLAGS.defense_type == 'defense_gan':

        recons_adv, zs = reconstructor.reconstruct(adv_x, batch_size=batch_size, reconstructor_id=123)

        preds_adv = model.get_logits(recons_adv)

        sess.run(tf.local_variables_initializer())

        diff_op = get_diff_op(model, adv_x, recons_adv, FLAGS.detect_image)
        z_norm = tf.reduce_sum(tf.square(zs), axis=1)

        acc_adv, diffs_mean, roc_info_adv = model_eval_gan(
            sess, images_pl, labels_pl, preds_adv, None,
            test_images=test_images, test_labels=test_labels, args=eval_params, diff_op=diff_op,
            z_norm=z_norm, recons_adv=recons_adv, adv_x=adv_x, debug=FLAGS.debug, vis_dir=_get_vis_dir(gan, FLAGS.attack_type))

        # reconstruction on clean images
        recons_clean, zs = reconstructor.reconstruct(images_pl_transformed, batch_size=batch_size)
        preds_eval = model.get_logits(recons_clean)

        sess.run(tf.local_variables_initializer())

        diff_op = get_diff_op(model, images_pl_transformed, recons_clean, FLAGS.detect_image)
        z_norm = tf.reduce_sum(tf.square(zs), axis=1)

        acc_rec, diffs_mean_rec, roc_info_rec = model_eval_gan(
            sess, images_pl, labels_pl, preds_eval, None,
            test_images=test_images, test_labels=test_labels, args=eval_params, diff_op=diff_op,
            z_norm=z_norm, recons_adv=recons_clean, adv_x=images_pl, debug=FLAGS.debug, vis_dir=_get_vis_dir(gan, 'clean'))

        # print('Training accuracy: {}'.format(train_acc))
        print('Evaluation accuracy: {}'.format(eval_acc))
        print('Evaluation accuracy with reconstruction: {}'.format(acc_rec))
        print('Test accuracy on adversarial examples: %0.4f\n' % acc_adv)

        return {'acc_adv': acc_adv,
                'acc_rec': acc_rec,
                'roc_info_adv': roc_info_adv,
                'roc_info_rec': roc_info_rec}
    else:
        preds_adv = model.get_logits(adv_x)
        sess.run(tf.local_variables_initializer())
        acc_adv = model_eval(sess, images_pl, labels_pl, preds_adv, test_images, test_labels,
                             args=eval_params)
        print('Test accuracy on adversarial examples: %0.4f\n' % acc_adv)

        return {'acc_adv': acc_adv,
                'acc_rec': 0,
                'roc_info_adv': None,
                'roc_info_rec': None}


import re


def main(cfg, argv=None):
    FLAGS = tf.app.flags.FLAGS

    tf.set_random_seed(11241990)
    np.random.seed(11241990)

    # Setting test time reconstruction hyper parameters.
    [tr_rr, tr_lr, tr_iters] = [FLAGS.rec_rr, FLAGS.rec_lr, FLAGS.rec_iters]

    gan = None
    if FLAGS.defense_type.lower() != 'none':
        if FLAGS.defense_type == 'defense_gan':
            gan = gan_from_config(cfg, True)

            gan.load_model()

            # Extract hyperparameters from reconstruction path.
            if FLAGS.rec_path is not None:
                train_param_re = re.compile('recs_rr(.*)_lr(.*)_iters(.*)')
                [tr_rr, tr_lr, tr_iters] = \
                    train_param_re.findall(FLAGS.rec_path)[0]
                gan.rec_rr = int(tr_rr)
                gan.rec_lr = float(tr_lr)
                gan.rec_iters = int(tr_iters)
            else:
                assert FLAGS.online_training or not FLAGS.train_on_recs

    if gan is None:
        gan = gan_from_config(cfg, True)
        gan.load_model()
        # gan = DefenseGANBase(cfg=cfg, test_mode=True)

    if FLAGS.override:
        gan.rec_rr = int(tr_rr)
        gan.rec_lr = float(tr_lr)
        gan.rec_iters = int(tr_iters)

    # Setting the results directory.
    results_dir, result_file_name = _get_results_dir_filename(gan)

    # Result file name. The counter ensures we are not overwriting the
    # results.
    counter = 0
    temp_fp = str(counter) + '_' + result_file_name
    results_dir = os.path.join(results_dir, FLAGS.results_dir)
    temp_final_fp = os.path.join(results_dir, temp_fp)
    while os.path.exists(temp_final_fp):
        counter += 1
        temp_fp = str(counter) + '_' + result_file_name
        temp_final_fp = os.path.join(results_dir, temp_fp)
    result_file_name = temp_fp
    sub_result_path = os.path.join(results_dir, result_file_name)

    accuracies = whitebox(
        gan, rec_data_path=FLAGS.rec_path,
        batch_size=FLAGS.batch_size,
        learning_rate=FLAGS.learning_rate,
        nb_epochs=FLAGS.nb_epochs,
        eps=FLAGS.fgsm_eps,
        online_training=FLAGS.online_training,
        defense_type=FLAGS.defense_type,
        num_tests=FLAGS.num_tests,
        attack_type=FLAGS.attack_type,
        num_train=FLAGS.num_train,
        cfg=cfg
    )

    ensure_dir(results_dir)

    with open(sub_result_path, 'a') as f:
        f.writelines([str(accuracies['acc_adv']) + ' ' + str(accuracies['acc_rec']) + '\n'])
        print('[*] saved accuracy in {}'.format(sub_result_path))

    if accuracies['roc_info_adv']:  # For attack detection.
        pkl_result_path = sub_result_path.replace('.txt', '_roc.pkl')
        with open(pkl_result_path, 'w') as f:
            cPickle.dump(accuracies['roc_info_adv'], f, cPickle.HIGHEST_PROTOCOL)
            print('[*] saved roc_info in {}'.format(pkl_result_path))

    if accuracies['roc_info_rec']:  # For attack detection.
        pkl_result_path = sub_result_path.replace('.txt', '_roc_clean.pkl')
        with open(pkl_result_path, 'w') as f:
            cPickle.dump(accuracies['roc_info_rec'], f, cPickle.HIGHEST_PROTOCOL)
            print('[*] saved roc_info_clean in {}'.format(pkl_result_path))


def _get_results_dir_filename(gan):
    FLAGS = tf.flags.FLAGS

    results_dir = os.path.join('results', 'whitebox_{}_{}'.format(
        FLAGS.defense_type, FLAGS.dataset_name))

    if FLAGS.defense_type == 'defense_gan':
        results_dir = gan.checkpoint_dir.replace('output', 'results')
        result_file_name = \
            'Iter={}_RR={:d}_LR={:.4f}_defense_gan'.format(
                gan.rec_iters,
                gan.rec_rr,
                gan.rec_lr,
                FLAGS.attack_type,
            )

        if not FLAGS.train_on_recs:
            result_file_name = 'orig_' + result_file_name
    elif FLAGS.defense_type == 'adv_tr':
        result_file_name = 'advTrEps={:.2f}'.format(FLAGS.fgsm_eps_tr)
    else:
        result_file_name = 'nodefense_'
    if FLAGS.num_tests > -1:
        result_file_name = 'numtest={}_'.format(
            FLAGS.num_tests) + result_file_name

    if FLAGS.num_train > -1:
        result_file_name = 'numtrain={}_'.format(
            FLAGS.num_train) + result_file_name

    if FLAGS.detect_image:
        result_file_name = 'det_image_' + result_file_name

    result_file_name = 'model={}_'.format(FLAGS.model) + result_file_name
    result_file_name += 'attack={}.txt'.format(FLAGS.attack_type)
    return results_dir, result_file_name


def _get_vis_dir(gan, attack_type):
    vis_dir = gan.checkpoint_dir.replace('output', 'vis')
    vis_dir = os.path.join(vis_dir, attack_type)
    return vis_dir


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', required=True, help='Config file')
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="RAND+FGSM random perturbation scale")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # Note: The load_config() call will convert all the parameters that are defined in
    # experiments/config files into FLAGS.param_name and can be passed in from command line.
    # arguments : python whitebox.py --cfg <config_path> --<param_name> <param_value>
    cfg = load_config(args.cfg)
    flags = tf.app.flags

    flags.DEFINE_integer('nb_classes', 10, 'Number of classes.')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training.')
    flags.DEFINE_integer('nb_epochs', 10, 'Number of epochs to train model.')
    flags.DEFINE_float('lmbda', 0.1, 'Lambda from arxiv.org/abs/1602.02697.')
    flags.DEFINE_float('fgsm_eps', 0.3, 'FGSM epsilon.')
    flags.DEFINE_string('rec_path', None, 'Path to reconstructions.')
    flags.DEFINE_integer('num_tests', -1, 'Number of test samples.')
    flags.DEFINE_integer('random_test_iter', -1,
                         'Number of random sampling for testing the classifier.')
    flags.DEFINE_boolean("online_training", False,
                         "Train the base classifier on reconstructions.")
    flags.DEFINE_string("defense_type", "none", "Type of defense [none|defense_gan|adv_tr]")
    flags.DEFINE_string("attack_type", "none", "Type of attack [fgsm|cw|bpda]")
    flags.DEFINE_integer("attack_iters", 100, 'Number of iterations for cw/pgd attack.')
    flags.DEFINE_integer("search_steps", 4, 'Number of binary search steps.')
    flags.DEFINE_string("results_dir", None, "The final subdirectory of the results.")
    flags.DEFINE_boolean("same_init", False, "Same initialization for z_hats.")
    flags.DEFINE_string("model", "F", "The classifier model.")
    flags.DEFINE_string("debug_dir", "temp", "The debug directory.")
    flags.DEFINE_integer("num_train", -1, 'Number of training data to load.')
    flags.DEFINE_boolean("debug", False, "True for saving reconstructions [False]")
    flags.DEFINE_boolean("load_classifier", False, "True for loading from saved classifier models [False]")
    flags.DEFINE_boolean("detect_image", False, "True for detection using image data [False]")
    flags.DEFINE_boolean("override", False, "Overriding the config values of reconstruction "
                                            "hyperparameters. It has to be true if either "
                                            "`--rec_rr`, `--rec_lr`, or `--rec_iters` is passed "
                                            "from command line.")
    flags.DEFINE_boolean("train_on_recs", False,
                         "Train the classifier on the reconstructed samples "
                         "using Defense-GAN.")

    main_cfg = lambda x: main(cfg, x)
    tf.app.run(main=main_cfg)
