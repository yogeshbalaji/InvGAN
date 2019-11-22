"""The building block ops for Spectral Normalization GAN."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


# tf.truncated_normal_initializer(stddev=0.02)
from tensorflow.python.ops.losses.losses_impl import Reduction

weight_init = tf.contrib.layers.xavier_initializer()
rng = np.random.RandomState([2016, 6, 1])

# adopted codes for BigGAN implementation
#   1. remove control dependency in spectral norm for cleaner interface
#   2. need to check BigGAN paper about weight regularization for generator
#   3. In the BigGAN paper, it is mentioned that adding BN in addition to SN cripples training


def conv2d(x, out_channels, kernel=3, stride=1, sn=False, update_collection=None, name='conv2d'):

    with tf.variable_scope(name):
        w = tf.get_variable('w', [kernel, kernel, x.get_shape()[-1], out_channels], initializer=weight_init)

        if sn:
            w = spectral_norm(w, update_collection=update_collection)

        conv = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME')

        bias = tf.get_variable('biases', [out_channels], initializer=tf.zeros_initializer())
        conv = tf.nn.bias_add(conv, bias)

        return conv


def deconv2d(x, out_channels, kernel=4, stride=2, sn=False, update_collection=None, name='deconv2d'):

    with tf.variable_scope(name):
        x_shape = x.get_shape().as_list()
        output_shape = [x_shape[0], x_shape[1]*stride, x_shape[2]*stride, out_channels]

        w = tf.get_variable('w', [kernel, kernel, out_channels, x_shape[-1]], initializer=weight_init)

        if sn:
            w = spectral_norm(w, update_collection=update_collection)

        deconv = tf.nn.conv2d_transpose(x, w, output_shape=output_shape, strides=[1, stride, stride, 1], padding='SAME')

        bias = tf.get_variable('biases', [out_channels], initializer=tf.zeros_initializer())
        deconv = tf.nn.bias_add(deconv, bias)
        deconv.shape.assert_is_compatible_with(output_shape)

        return deconv


def linear(x, out_features, sn=False, update_collection=None, name='linear'):

    with tf.variable_scope(name):
        x_shape = x.get_shape().as_list()
        assert (len(x_shape) == 2)

        matrix = tf.get_variable('W', [x_shape[1], out_features], tf.float32, initializer=weight_init)

        if sn:
            matrix = spectral_norm(matrix, update_collection=update_collection)

        bias = tf.get_variable('bias', [out_features], initializer=tf.zeros_initializer())
        out = tf.matmul(x, matrix) + bias
        return out


def embedding(labels, number_classes, embedding_size, update_collection=None, name='snembedding'):
    with tf.variable_scope(name):
        embedding_map = tf.get_variable(
            name='embedding_map',
            shape=[number_classes, embedding_size],
            initializer=tf.contrib.layers.xavier_initializer())

        embedding_map_bar_transpose = spectral_norm(tf.transpose(embedding_map), update_collection=update_collection)
        embedding_map_bar = tf.transpose(embedding_map_bar_transpose)

        return tf.nn.embedding_lookup(embedding_map_bar, labels)

##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.tanh(x)


##################################################################################
# Sampling
##################################################################################

def global_sum_pooling(x):
    gsp = tf.reduce_sum(x, axis=[1, 2])
    return gsp


def up_sample(x):
    _, h, w, _ = x.get_shape().as_list()
    x = tf.image.resize_nearest_neighbor(x, [h * 2, w * 2])
    return x


def down_sample(x):
    x = tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
    return x


##################################################################################
# Normalization
##################################################################################

def batch_norm(x, is_training=True, name='batch_norm'):
    return tf.contrib.layers.batch_norm(x, decay=0.9, epsilon=1e-05,
                                        center=True, scale=True,
                                        is_training=is_training, scope=name, updates_collections=None)


def condition_batch_norm(x, z, is_training=True, scope='batch_norm'):
    """
    Hierarchical Embedding (without class-conditioning).
    Input latent vector z is linearly projected to produce per-sample gain and bias for batchnorm

    Note: Each instance has (2 x len(z) x n_feature) parameters
    """
    with tf.variable_scope(scope):
        _, _, _, c = x.get_shape().as_list()
        decay = 0.9
        epsilon = 1e-05

        test_mean = tf.get_variable("pop_mean", shape=[c], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=False)
        test_var = tf.get_variable("pop_var", shape=[c], dtype=tf.float32, initializer=tf.constant_initializer(1.0), trainable=False)

        beta = linear(z, c, name='beta')
        gamma = linear(z, c, name='gamma')

        beta = tf.reshape(beta, shape=[-1, 1, 1, c])
        gamma = tf.reshape(gamma, shape=[-1, 1, 1, c])

        if is_training:
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
            ema_mean = tf.assign(test_mean, test_mean * decay + batch_mean * (1 - decay))
            ema_var = tf.assign(test_var, test_var * decay + batch_var * (1 - decay))

            with tf.control_dependencies([ema_mean, ema_var]):
                return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, epsilon)
        else:
            return tf.nn.batch_normalization(x, test_mean, test_var, beta, gamma, epsilon)


class ConditionalBatchNorm(object):
    """Conditional BatchNorm.
    For each  class, it has a specific gamma and beta as normalization variable.

    Note: Each batch norm has (2 x n_class x n_feature) parameters
    """

    def __init__(self, num_categories, name='conditional_batch_norm', decay_rate=0.999, center=True,
               scale=True):
        with tf.variable_scope(name):
            self.name = name
            self.num_categories = num_categories
            self.center = center
            self.scale = scale
            self.decay_rate = decay_rate

    def __call__(self, inputs, labels, is_training=True):
        inputs = tf.convert_to_tensor(inputs)
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
        #axis = [0, 1, 2]
        axis = range(0, len(inputs_shape)-1)
        shape = tf.TensorShape([self.num_categories]).concatenate(params_shape)
        #moving_shape = tf.TensorShape([1, 1, 1]).concatenate(params_shape)
        moving_shape = tf.TensorShape((len(inputs_shape)-1)*[1]).concatenate(params_shape)

        with tf.variable_scope(self.name):
            self.gamma = tf.get_variable(
                'gamma', shape,
                initializer=tf.ones_initializer())
            self.beta = tf.get_variable(
                'beta', shape,
                initializer=tf.zeros_initializer())
            self.moving_mean = tf.get_variable('mean', moving_shape,
                              initializer=tf.zeros_initializer(),
                              trainable=False)
            self.moving_var = tf.get_variable('var', moving_shape,
                              initializer=tf.ones_initializer(),
                              trainable=False)

            beta = tf.gather(self.beta, labels)
            gamma = tf.gather(self.gamma, labels)

            for _ in range(len(inputs_shape) - len(shape)):
                beta = tf.expand_dims(beta, 1)
                gamma = tf.expand_dims(gamma, 1)

            decay = self.decay_rate
            variance_epsilon = 1E-5
            if is_training:
                mean, variance = tf.nn.moments(inputs, axis, keep_dims=True)
                update_mean = tf.assign(self.moving_mean, self.moving_mean * decay + mean * (1 - decay))
                update_var = tf.assign(self.moving_var, self.moving_var * decay + variance * (1 - decay))
                #tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mean)
                #tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_var)
                with tf.control_dependencies([update_mean, update_var]):
                    outputs = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, variance_epsilon)
            else:
                outputs = tf.nn.batch_normalization(
                      inputs, self.moving_mean, self.moving_var, beta, gamma, variance_epsilon)
            outputs.set_shape(inputs_shape)
            return outputs


def _l2normalize(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def spectral_norm(w, num_iters=1, update_collection=None):
    """
    https://github.com/taki0112/BigGAN-Tensorflow/blob/master/ops.py
    """
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable('u', [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for _ in range(num_iters):
        v_ = tf.matmul(u_hat, w, transpose_b=True)
        v_hat = _l2normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = _l2normalize(u_)

    sigma = tf.squeeze(tf.matmul(tf.matmul(v_hat, w), u_hat, transpose_b=True))
    w_norm = w / sigma

    if update_collection is None:
        with tf.control_dependencies([u.assign(u_hat)]):
            w_norm = tf.reshape(w_norm, w_shape)
    elif update_collection == 'NO_OPS':
        w_norm = tf.reshape(w_norm, w_shape)
    else:
        raise NotImplementedError

    return w_norm


##################################################################################
# Residual Blockes
##################################################################################

def resblock_up(x, out_channels, is_training=True, sn=False, update_collection=None, name='resblock_up'):
    with tf.variable_scope(name):
        x_0 = x
        # block 1
        x = tf.nn.relu(batch_norm(x, is_training=is_training, name='bn1'))
        x = up_sample(x)
        x = conv2d(x, out_channels, 3, 1, sn=sn, update_collection=update_collection, name='conv1')

        # block 2
        x = tf.nn.relu(batch_norm(x, is_training=is_training, name='bn2'))
        x = conv2d(x, out_channels, 3, 1, sn=sn, update_collection=update_collection, name='conv2')

        # skip connection
        x_0 = up_sample(x_0)
        x_0 = conv2d(x_0, out_channels, 1, 1, sn=sn, update_collection=update_collection, name='conv3')

        return x_0 + x


def resblock_down(x, out_channels, sn=False, update_collection=None, downsample=True, name='resblock_down'):
    with tf.variable_scope(name):
        input_channels = x.shape.as_list()[-1]
        x_0 = x
        x = tf.nn.relu(x)
        x = conv2d(x, out_channels, 3, 1, sn=sn, update_collection=update_collection, name='sn_conv1')
        x = tf.nn.relu(x)
        x = conv2d(x, out_channels, 3, 1, sn=sn, update_collection=update_collection, name='sn_conv2')

        if downsample:
            x = down_sample(x)
        if downsample or input_channels != out_channels:
            x_0 = conv2d(x_0, out_channels, 1, 1, sn=sn, update_collection=update_collection, name='sn_conv3')
            if downsample:
                x_0 = down_sample(x_0)

        return x_0 + x


def inblock(x, out_channels, sn=False, update_collection=None, name='inblock'):
    with tf.variable_scope(name):
        x_0 = x
        x = conv2d(x, out_channels, 3, 1, sn=sn, update_collection=update_collection, name='sn_conv1')
        x = tf.nn.relu(x)
        x = conv2d(x, out_channels, 3, 1, sn=sn, update_collection=update_collection, name='sn_conv2')

        x = down_sample(x)
        x_0 = down_sample(x_0)
        x_0 = conv2d(x_0, out_channels, 1, 1, sn=sn, update_collection=update_collection, name='sn_conv3')

        return x_0 + x


##################################################################################
# Loss Functions
##################################################################################

def discriminator_loss(loss_func, real, fake):

    real_loss = 0
    fake_loss = 0

    if loss_func.__contains__('wgan'):
        real_loss = -tf.reduce_mean(real)
        fake_loss = tf.reduce_mean(fake)

    if loss_func == 'dcgan':
        real_loss = tf.losses.sigmoid_cross_entropy(
            tf.ones_like(real), real, reduction=Reduction.MEAN,
        )
        fake_loss = tf.losses.sigmoid_cross_entropy(
            tf.zeros_like(fake), fake, reduction=Reduction.MEAN,
        )

    if loss_func == 'hingegan':
        real_loss = tf.reduce_mean(relu(1 - real))
        fake_loss = tf.reduce_mean(relu(1 + fake))

    if loss_func == 'ragan':
        real_loss = tf.reduce_mean(tf.nn.softplus(-(real - tf.reduce_mean(fake))))
        fake_loss = tf.reduce_mean(tf.nn.softplus(fake - tf.reduce_mean(real)))

    loss = real_loss + fake_loss

    return loss


def generator_loss(loss_func, fake):
    fake_loss = 0

    if loss_func.__contains__('wgan'):
        fake_loss = -tf.reduce_mean(fake)

    if loss_func == 'dcgan':
        fake_loss = tf.losses.sigmoid_cross_entropy(
            fake, tf.ones_like(fake), reduction=Reduction.MEAN,
        )

    if loss_func == 'hingegan':
        fake_loss = -tf.reduce_mean(fake)

    return fake_loss


def encoder_gan_loss(loss_func, fake):
    fake_loss = 0

    if loss_func.__contains__('wgan'):
        fake_loss = -tf.reduce_mean(fake)

    if loss_func == 'dcgan':
        fake_loss = tf.reduce_mean(tf.nn.softplus(-fake))

    if loss_func == 'hingegan':
        fake_loss = -tf.reduce_mean(fake)

    return fake_loss













