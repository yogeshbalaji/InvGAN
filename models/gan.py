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

"""Contains the GAN implementations of the abstract model class."""

import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

import tflib
import tflib.cifar10
import tflib.mnist
import tflib.plot
import tflib.save_images
from models.dataset_networks import get_generator_fn, get_discriminator_fn, \
    get_encoder_fn
from tflib.layers import generator_loss, discriminator_loss
from datasets.utils import get_generators
from models.base_model import AbstractModel

from utils.misc import ensure_dir
from utils.visualize import save_images_files


class DefenseGANBase(AbstractModel):
    @property
    def default_properties(self):
        return [
            'dataset_name', 'batch_size', 'use_bn', 'use_resblock',
            'test_batch_size',
            'mode', 'gradient_penalty_lambda', 'train_iters', 'enc_train_iters',
            'critic_iters', 'latent_dim', 'net_dim',
            'input_transform_type',
            'debug', 'rec_iters', 'image_dim', 'rec_rr',
            'rec_lr', 'test_again', 'loss_type',
            'attribute', 'encoder_loss_type', 'use_encoder_init',
            'encoder_lr', 'discriminator_lr', 'generator_lr',
            'discriminator_rec_lr']

    def __init__(self, cfg=None, test_mode=False, verbose=True, **args):
        self.dataset_name = None  # Name of the datsaet.
        self.batch_size = 32  # Batch size for training the GAN.
        self.use_bn = True  # Use batchnorm in the discriminator and generator.
        self.use_resblock = False  # Use resblocks in DefenseGAN.
        self.test_batch_size = 20  # Batch size for test time.
        self.mode = 'wgan-gp'  # The mode of training the GAN (default: gp-wgan).
        self.gradient_penalty_lambda = 10.0  # Gradient penalty scale.
        self.train_iters = 200000  # Number of training iterations.
        self.enc_train_iters = 100000
        self.critic_iters = 5  # Critic iterations per training step.
        self.latent_dim = None  # The dimension of the latent vectors.
        self.net_dim = None  # The complexity of network per layer.
        self.input_transform_type = 0  # The normalization used for the inputs.
        self.debug = False  # Debug info will be printed.
        self.rec_iters = 200  # Number of reconstruction iterations.
        self.image_dim = [None, None, None]  # [height, width, number of channels] of the output image.
        self.rec_rr = 10  # Number of random restarts for the reconstruction
        self.encoder_loss_type = 'margin' # Loss used for encoder
        self.use_encoder_init = True # Flag for using encoder initialization while reconstructing

        self.rec_lr = 10.0  # The reconstruction learning rate.
        self.test_again = False  # If true, do not use the cached info for test phase.
        self.attribute = 'gender'

        self.encoder_lr = 2e-4
        self.generator_lr = 1e-4
        self.discriminator_lr = 4e-4
        self.discriminator_rec_lr = 4e-4

        # Should be implemented in the child classes.
        self.discriminator_fn = None
        self.generator_fn = None
        self.encoder_fn = None
        self.train_data_gen = None

        self.model_save_name = 'GAN.model'

        # calls _build() and _loss()
        # generator_vars and encoder_vars are created
        super(DefenseGANBase, self).__init__(
                                             test_mode=test_mode,
                                             verbose=verbose, cfg=cfg, **args)
        self.save_var_prefixes = ['Generator', 'Discriminator']
        self._load_dataset()

        # create a method that only loads generator and encoder
        g_saver = tf.train.Saver(var_list=self.generator_vars)
        self.load_generator = lambda ckpt_path=None: self.load(checkpoint_dir=ckpt_path, saver=g_saver)

        d_saver = tf.train.Saver(var_list=self.discriminator_vars)
        self.load_discriminator = lambda ckpt_path=None: self.load(checkpoint_dir=ckpt_path, saver=d_saver)

    def _build_generator_discriminator(self):
        """Creates the generator and discriminator graph per dataset."""
        discriminator_fn = get_discriminator_fn(self.dataset_name, use_resblock=self.use_resblock)
        self.discriminator_fn = discriminator_fn

        generator_fn = get_generator_fn(self.dataset_name, use_resblock=self.use_resblock)
        self.generator_fn = lambda x, is_training=self.is_training: generator_fn(x, is_training=is_training)

    def _load_dataset(self):
        """Loads the dataset."""
        self.train_data_gen, self.dev_gen, _ = get_generators(self.dataset_name, self.batch_size)
        self.train_gen_test, self.dev_gen_test, self.test_gen_test = \
            get_generators(self.dataset_name, self.test_batch_size, randomize=False)

    def _build(self):
        """Builds the computation graph."""

        assert (self.batch_size % self.rec_rr) == 0, 'Batch size should be divisable by random restart'

        self.test_batch_size = self.batch_size

        # Defining batch_size in input placeholders is inevitable at least
        # for now, because the z vectors are Tensorflow variables.
        self.real_data_pl = tf.placeholder(
            tf.float32, shape=[self.batch_size] + self.image_dim,
        )
        self.real_data_test_pl = tf.placeholder(
            tf.float32, shape=[self.test_batch_size] + self.image_dim,
        )
        z = tf.random_normal([self.batch_size, self.latent_dim])

        self.input_pl_transform()
        self._build_generator_discriminator()

        self.fake_data = self.generator_fn(z, is_training=self.is_training)
        self.disc_real = self.discriminator_fn(self.real_data)
        self.disc_fake = self.discriminator_fn(self.fake_data, update_collection='NO_OPS')

        self.fixed_noise = tf.constant(
            np.random.normal(size=(128, self.latent_dim)).astype('float32'))
        self.fixed_noise_samples = self.generator_fn(self.fixed_noise, is_training=False)

        # variables for saving and loading (e.g. batchnorm moving average)
        self.generator_vars = slim.get_variables('Generator')
        self.discriminator_vars = slim.get_variables('Discriminator')

        # trainable variables
        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

    def _loss(self):
        """Builds the loss part of the graph.."""
        self.encoder_cost = 0
        self.generator_cost = generator_loss(self.mode, self.disc_fake)
        self.discriminator_cost = discriminator_loss(self.mode, self.disc_real, self.disc_fake)
        self.clip_disc_weights = None

        if self.mode == 'wgan':
            clip_ops = []
            for var in tflib.params_with_name('Discriminator'):
                clip_bounds = [-.01, .01]
                clip_ops.append(
                    tf.assign(var, tf.clip_by_value(var, clip_bounds[0], clip_bounds[1]))
                )
            self.clip_disc_weights = tf.group(*clip_ops)

        elif self.mode == 'wgan-gp':
            alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1)
            differences = self.fake_data - self.real_data
            interpolates = self.real_data + (alpha * differences)
            gradients = tf.gradients(self.discriminator_fn(interpolates), [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            self.discriminator_cost += self.gradient_penalty_lambda * gradient_penalty

        # define optimizer op
        self.gen_train_op = tf.train.AdamOptimizer(
            learning_rate=self.generator_lr,
            beta1=0.5).minimize(self.generator_cost, var_list=self.g_vars)
        self.disc_train_op = tf.train.AdamOptimizer(
            learning_rate=self.discriminator_lr,
            beta1=0.5).minimize(self.discriminator_cost, var_list=self.d_vars)

        # summary writer
        g_loss_summary_op = tf.summary.scalar('g_loss', self.generator_cost)
        d_loss_summary_op = tf.summary.scalar('d_loss', self.discriminator_cost)
        self.merged_summary_op = tf.summary.merge_all()

    def _inf_train_gen(self):
        """A generator function for input training data."""
        while True:
            for images, targets in self.train_data_gen():
                yield images

    def load_model(self):
        could_load_generator = self.load_generator(ckpt_path=self.checkpoint_dir)
        assert could_load_generator
        self.initialized = True

    def train(self, phase=None):
        """Trains the GAN model."""

        sess = self.sess
        self.initialize_uninitialized()

        gen = self._inf_train_gen()
        # (1) initialize saver
        # (2) load checkpoint if any
        could_load = self.load(checkpoint_dir=self.checkpoint_dir, prefixes=self.save_var_prefixes)
        if could_load:
            print('[*] Model loaded.')
        else:
            print('[#] No model found')

        cur_iter = self.sess.run(self.global_step)
        max_train_iters = self.train_iters
        step_inc = self.global_step_inc
        global_step = self.global_step
        ckpt_dir = self.checkpoint_dir

        for iteration in xrange(cur_iter, max_train_iters):
            start_time = time.time()

            _ = sess.run(self.gen_train_op, feed_dict={self.is_training: True, self.is_training_enc: False})

            for i in xrange(self.critic_iters):
                _data = gen.next()
                _disc_cost, _, summaries = sess.run(
                    [self.discriminator_cost, self.disc_train_op, self.merged_summary_op],
                    feed_dict={self.real_data_pl: _data, self.is_training: True, self.is_training_enc: False})
                if self.clip_disc_weights is not None:
                    _ = sess.run(self.clip_disc_weights)

            tflib.plot.plot('{}/train disc cost'.format(self.debug_dir), _disc_cost)
            tflib.plot.plot('{}/time'.format(self.debug_dir), time.time() - start_time)

            # Calculate dev loss and generate samples every 500 iters.
            if iteration % 500 == 5:
                dev_disc_costs = []
                dev_ctr = 0
                for images, _ in self.dev_gen():
                    dev_ctr += 1
                    if dev_ctr > 20:
                        break
                    _dev_disc_cost = sess.run(
                        self.discriminator_cost,
                        feed_dict={self.real_data_pl: images,
                                   self.is_training: False,
                                   self.is_training_enc: False}
                    )
                    dev_disc_costs.append(_dev_disc_cost)
                tflib.plot.plot('{}/dev disc cost'.format(self.debug_dir), np.mean(dev_disc_costs))
                self.generate_image(iteration)
                self.summary_writer.add_summary(summaries, global_step=iteration)

            # Write logs every 100 iters
            if (iteration < 5) or (iteration % 100 == 99):
                tflib.plot.flush()

            self.sess.run(step_inc)
            if iteration % 500 == 499:
                self.save(checkpoint_dir=ckpt_dir, global_step=global_step)

            tflib.plot.tick()

        self.save(checkpoint_dir=ckpt_dir, global_step=global_step)

    def train_encoder(self, gan_init_path=None):
        sess = self.sess
        self.initialize_uninitialized()
        self.save_var_prefixes = ['Generator', 'Encoder']

        gen = self._inf_train_gen()

        # (1) load generator
        # (2) self.saver will be initialized to save generator and encoder
        # potential bug if self.saver is already initialized. e.g. train() is called before train_encoder()
        if gan_init_path is None:
            gan_init_path = self.checkpoint_dir

        could_load = self.load_generator(gan_init_path) and self.load_discriminator(gan_init_path)

        if could_load:
            print('[*] Model loaded.')
        else:
            print('[#] No model found')

        cur_iter = self.sess.run(self.global_step)
        max_train_iters = self.enc_train_iters
        step_inc = self.global_step_inc
        global_step = self.global_step
        ckpt_dir = self.encoder_checkpoint_dir

        # sanity check for the generator
        samples = self.sess.run(self.fixed_noise_samples, feed_dict={self.is_training: False})
        self.save_image(samples, 'sanity_check.png')

        for iteration in xrange(cur_iter, max_train_iters):
            start_time = time.time()
            _data = gen.next()

            # Discriminator update
            _ = sess.run([self.disc_rec_train_op],
                               feed_dict={self.real_data_pl: _data, self.is_training_enc: True, self.is_training: True})

            # Encoder update
            loss, _ = sess.run([self.encoder_cost, self.encoder_train_op],
                               feed_dict={self.real_data_pl: _data, self.is_training_enc: True, self.is_training: True})

            tflib.plot.plot('{}/train encoder cost'.format(self.debug_dir), loss)
            tflib.plot.plot('{}/time'.format(self.debug_dir), time.time() - start_time)

            if (iteration < 5) or (iteration % 100 == 99):
                tflib.plot.flush()

            self.sess.run(step_inc)

            if iteration % 1000 == 999:
                x_hat_sample, x_hat, x = sess.run([self.x_hat_sample, self.x_hat, self.real_data],
                                    feed_dict={self.real_data_pl: _data, self.is_training: False,
                                    self.is_training_enc: False})
                self.save_image(x_hat_sample, 'x_hat_sample_{}.png'.format(iteration))
                self.save_image(x_hat, 'x_hat_{}.png'.format(iteration))
                self.save_image(x, 'x_{}.png'.format(iteration))
                self.save(checkpoint_dir=ckpt_dir, global_step=global_step)

            tflib.plot.tick()

        self.save(checkpoint_dir=ckpt_dir, global_step=global_step)

    def reconstruct(self, images, batch_size=None, back_prop=False, reconstructor_id=0, z_init_val=None):
        """Creates the reconstruction op for Defense-GAN.

        Args:
            X: Input tensor

        Returns:
            The `tf.Tensor` of the reconstructed input.
        """

        # Batch size is needed because the latent codes are `tf.Variable`s and
        # need to be built into TF's static graph beforehand.

        batch_size = batch_size if batch_size else self.test_batch_size

        x_shape = images.get_shape().as_list()
        x_shape[0] = batch_size

        # Repeat images self.rec_rr times to handle random restarts in
        # parallel.
        images_tiled_rr = tf.reshape(
            images, [x_shape[0], np.prod(x_shape[1:])])
        images_tiled_rr = tf.tile(images_tiled_rr, [1, self.rec_rr])
        images_tiled_rr = tf.reshape(
            images_tiled_rr, [x_shape[0] * self.rec_rr] + x_shape[1:])

        # Number of reconstruction iterations.
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            rec_iter_const = tf.get_variable(
                'rec_iter_{}'.format(reconstructor_id),
                initializer=tf.constant(0),
                trainable=False, dtype=tf.int32,
                collections=[tf.GraphKeys.LOCAL_VARIABLES],
            )
            # The latent variables.
            z_hat = tf.get_variable(
                'z_hat_rec_{}'.format(reconstructor_id),
                shape=[batch_size * self.rec_rr, self.latent_dim],
                dtype=tf.float32,
                initializer=tf.random_normal_initializer(
                    stddev=np.sqrt(1.0 / self.latent_dim)),
                collections=[tf.GraphKeys.LOCAL_VARIABLES]
            )

        # Learning rate for reconstruction.
        rec_lr_op_from_const = self.get_learning_rate(init_lr=self.rec_lr,
                                                      global_step=rec_iter_const,
                                                      decay_mult=0.1,
                                                      decay_iter=np.ceil(
                                                          self.rec_iters *
                                                          0.8).astype(
                                                          np.int32))

        # The optimizer.
        rec_online_optimizer = tf.train.MomentumOptimizer(
            learning_rate=rec_lr_op_from_const, momentum=0.7,
            name='rec_optimizer')

        init_z = tf.assign(z_hat, tf.constant(np.random.randn(batch_size * self.rec_rr, self.latent_dim).astype(np.float32)))

        if z_init_val is not None:
            # Repeat z_init_val for rec_rr times
            z_shape = z_init_val.get_shape().as_list()
            z_shape[0] = batch_size

            z_tiled_rr = tf.tile(z_init_val, [1, self.rec_rr])
            z_tiled_rr = tf.reshape(
                z_tiled_rr, [z_shape[0] * self.rec_rr] + z_shape[1:])
            init_z = tf.assign(z_hat, z_tiled_rr)

        z_hats_recs = self.generator_fn(z_hat, is_training=False)
        num_dim = len(z_hats_recs.get_shape())
        axes = range(1, num_dim)

        image_rec_loss = tf.reduce_mean(
            tf.square(z_hats_recs - images_tiled_rr),
            axis=axes)
        rec_loss = tf.reduce_sum(image_rec_loss)
        rec_online_optimizer.minimize(rec_loss, var_list=[z_hat])

        def rec_body(i, *args):
            z_hats_recs = self.generator_fn(z_hat, is_training=False)
            image_rec_loss = tf.reduce_mean(
                tf.square(z_hats_recs - images_tiled_rr),
                axis=axes)
            rec_loss = tf.reduce_sum(image_rec_loss)

            train_op = rec_online_optimizer.minimize(rec_loss,
                                                     var_list=[z_hat])

            return tf.tuple(
                [tf.add(i, 1), rec_loss, image_rec_loss, z_hats_recs, z_hat],
                control_inputs=[train_op])

        rec_iter_condition = lambda i, *args: tf.less(i, self.rec_iters)
        for opt_var in rec_online_optimizer.variables():
            tf.add_to_collection(
                tf.GraphKeys.LOCAL_VARIABLES,
                opt_var,
            )

        with tf.control_dependencies([init_z]):
            online_rec_iter, online_rec_loss, online_image_rec_loss, \
            all_z_recs, all_zs = tf.while_loop(
                rec_iter_condition,
                rec_body,
                [rec_iter_const, rec_loss, image_rec_loss, z_hats_recs, z_hat]
                , parallel_iterations=1, back_prop=back_prop,
                swap_memory=False)
            final_recs = []
            final_zs = []
            for i in range(batch_size):
                ind = i * self.rec_rr + tf.argmin(
                    online_image_rec_loss[
                    i * self.rec_rr:(i + 1) * self.rec_rr
                    ],
                    axis=0)
                final_recs.append(all_z_recs[tf.cast(ind, tf.int32)])
                final_zs.append(all_zs[tf.cast(ind, tf.int32)])

            online_rec = tf.stack(final_recs)
            online_zs = tf.stack(final_zs)

            return tf.stop_gradient(tf.reshape(online_rec, x_shape)), tf.stop_gradient(online_zs)

    def generate_image(self, iteration=None):
        """Generates a fixed noise for visualization of generation output.
        """
        samples = self.sess.run(self.fixed_noise_samples, feed_dict={self.is_training: False})
        tflib.save_images.save_images(self.imsave_transform(samples),
                                      os.path.join(self.checkpoint_dir.replace('output', 'debug'),
                                                   'samples_{}.png'.format(iteration)))

    def save_image(self, images, name):
        tflib.save_images.save_images(self.imsave_transform(images), os.path.join(self.encoder_debug_dir, name))

    def test_batch(self):
        """Tests the image batch generator."""
        output_dir = os.path.join(self.debug_dir, 'test_batch')
        ensure_dir(output_dir)

        img, target = self.train_data_gen().next()
        img = img.reshape([self.batch_size] + self.image_dim)
        save_images_files(img / 255.0, output_dir=output_dir, labels=target)
