import tensorflow as tf
from tflib.layers import *


def mnist_generator(z, is_training=True):
    net_dim = 64
    use_sn = False
    update_collection = None
    with tf.variable_scope('Generator', reuse=tf.AUTO_REUSE):
        output = linear(z, 4*4*4*net_dim, sn=use_sn, name='linear')
        output = batch_norm(output, is_training=is_training, name='bn_linear')
        output = tf.nn.relu(output)
        output = tf.reshape(output, [-1, 4, 4, 4*net_dim])
        
        # deconv-bn-relu
        output = deconv2d(output, 2*net_dim, 5, 2, sn=use_sn, name='deconv_0')
        output = batch_norm(output, is_training=is_training, name='bn_0')
        output = tf.nn.relu(output)

        output = output[:, :7, :7, :]

        output = deconv2d(output, net_dim, 5, 2, sn=use_sn, name='deconv_1')
        output = batch_norm(output, is_training=is_training, name='bn_1')
        output = tf.nn.relu(output)

        output = deconv2d(output, 1, 5, 2, sn=use_sn, name='deconv_2')
        output = tf.sigmoid(output)

        return output


def mnist_discriminator(x, update_collection=None, is_training=False):
    net_dim = 64
    use_sn = True
    with tf.variable_scope('Discriminator', reuse=tf.AUTO_REUSE):
        # block 1
        x = conv2d(x, net_dim, 5, 2, sn=use_sn, update_collection=update_collection, name='conv0')
        x = lrelu(x)
        # block 2
        x = conv2d(x, 2 * net_dim, 5, 2, sn=use_sn, update_collection=update_collection, name='conv1')
        x = lrelu(x)
        # block 3
        x = conv2d(x, 4 * net_dim, 5, 2, sn=use_sn, update_collection=update_collection, name='conv2')
        x = lrelu(x)
        # output
        x = tf.reshape(x, [-1, 4 * 4 * 4 * net_dim])
        x = linear(x, 1, sn=use_sn, update_collection=update_collection, name='linear')
        return tf.reshape(x, [-1])


def mnist_encoder(x, is_training=False, use_bn=False, net_dim=64, latent_dim=128):
    with tf.variable_scope('Encoder', reuse=tf.AUTO_REUSE):
        x = conv2d(x, net_dim, 5, 2, name='conv0')
        if use_bn:
            x = batch_norm(x, is_training=is_training, name='bn0')
        x = tf.nn.relu(x)

        x = conv2d(x, 2*net_dim, 5, 2, name='conv1')
        if use_bn:
            x = batch_norm(x, is_training=is_training, name='bn1')
        x = tf.nn.relu(x)

        x = conv2d(x, 4*net_dim, 5, 2, name='conv2')
        if use_bn:
            x = batch_norm(x, is_training=is_training, name='bn2')
        x = tf.nn.relu(x)

        x = tf.reshape(x, [-1, 4 * 4 * 4 * net_dim])
        x = linear(x, 2*latent_dim, name='linear')

        return x[:, :latent_dim], x[:, latent_dim:]


def cifar10_generator_resnet(z, is_training=False):
    with tf.variable_scope('Generator', reuse=tf.AUTO_REUSE):
        output = linear(z, 4 * 4 * 256, name='linear_0')
        output = tf.reshape(output, [-1, 4, 4, 256])
        output = resblock_up(output, 256, is_training=is_training, name='block_0')
        output = resblock_up(output, 256, is_training=is_training, name='block_1')
        output = resblock_up(output, 256, is_training=is_training, name='block_2')

        # output layers
        output = batch_norm(output, is_training=is_training, name='g_bn')
        output = tf.nn.relu(output)
        output = conv2d(output, 3, 3, 1, name='conv_last')
        output = tf.tanh(output)

        return output


def cifar10_discriminator_resnet(x, update_collection=None, is_training=False):
    with tf.variable_scope('Discriminator', reuse=tf.AUTO_REUSE):
        output = inblock(x, 128, sn=True, update_collection=update_collection, name='block_0') # 16 x 16
        output = resblock_down(output, 128, sn=True, update_collection=update_collection, name='block_1') # 8 x 8
        output = resblock_down(output, 128, sn=True, update_collection=update_collection, downsample=False, name='block_2')
        output = resblock_down(output, 128, sn=True, update_collection=update_collection, downsample=False, name='block_3')

        # output layers
        output = tf.nn.relu(output)
        output = tf.reduce_sum(output, [1, 2])
        output = linear(output, 1, sn=True, update_collection=update_collection, name='linear')

        return tf.reshape(output, [-1])


def cifar10_generator(z, is_training=False):
    net_dim = 64
    use_sn = False
    update_collection = None
    with tf.variable_scope('Generator', reuse=tf.AUTO_REUSE):
        output = linear(z, 4*4*8*net_dim, sn=use_sn, name='linear')
        output = batch_norm(output, is_training=is_training, name='bn_linear')
        output = tf.nn.relu(output)
        output = tf.reshape(output, [-1, 4, 4, 8*net_dim])

        # deconv-bn-relu
        for i in range(3):
            output = deconv2d(output, 2**(2-i)*net_dim, sn=use_sn, name='deconv_' + str(i))
            output = batch_norm(output, is_training=is_training, name='bn_' + str(i))
            output = tf.nn.relu(output)
        # conv
        output = conv2d(output, 3, sn=use_sn, name='conv3')
        output = tf.tanh(output)

        return output


def cifar10_discriminator(x, update_collection=None, is_training=False):
    net_dim = 64
    use_sn = True
    with tf.variable_scope('Discriminator', reuse=tf.AUTO_REUSE):
        # block 1
        x = conv2d(x, net_dim, 3, 1, sn=use_sn, update_collection=update_collection, name='conv0')
        x = lrelu(x)
        x = conv2d(x, net_dim, 4, 2, sn=use_sn, update_collection=update_collection, name='conv1')
        x = lrelu(x)
        # block 2
        x = conv2d(x, 2*net_dim, 3, 1, sn=use_sn, update_collection=update_collection, name='conv2')
        x = lrelu(x)
        x = conv2d(x, 2*net_dim, 4, 2, sn=use_sn, update_collection=update_collection, name='conv3')
        x = lrelu(x)
        # block 3
        x = conv2d(x, 4*net_dim, 3, 1, sn=use_sn, update_collection=update_collection, name='conv4')
        x = lrelu(x)
        x = conv2d(x, 4*net_dim, 4, 2, sn=use_sn, update_collection=update_collection, name='conv5')
        x = lrelu(x)
        # output
        x = conv2d(x, 8*net_dim, 3, 1, sn=use_sn, update_collection=update_collection, name='conv6')
        x = tf.reshape(x, [-1, 4*4*8*net_dim])
        x = linear(x, 1, sn=use_sn, update_collection=update_collection, name='linear')
        return tf.reshape(x, [-1])


def cifar10_encoder(x, is_training=False, use_bn=False, net_dim=64, latent_dim=128):
    with tf.variable_scope('Encoder', reuse=tf.AUTO_REUSE):
        x = conv2d(x, net_dim, 3, 1, name='conv0')
        if use_bn:
            x = batch_norm(x, is_training=is_training, name='bn0')
        x = tf.nn.relu(x)

        x = conv2d(x, 2*net_dim, 4, 2, name='conv1')
        if use_bn:
            x = batch_norm(x, is_training=is_training, name='bn1')
        x = tf.nn.relu(x)

        x = conv2d(x, 4*net_dim, 4, 2, name='conv2')
        if use_bn:
            x = batch_norm(x, is_training=is_training, name='bn2')
        x = tf.nn.relu(x)

        x = conv2d(x, 8*net_dim, 4, 2, name='conv3')
        if use_bn:
            x = batch_norm(x, is_training=is_training, name='bn3')
        x = tf.nn.relu(x)

        x = tf.reshape(x, [-1, 4 * 4 * 8 * net_dim])
        x = linear(x, 2*latent_dim, name='linear')

        return x[:, :latent_dim], x[:, latent_dim:]


def cifar10_encoder_resnet(x, is_training=False, use_bn=False, net_dim=64, latent_dim=128):
    with tf.variable_scope('Encoder', reuse=tf.AUTO_REUSE):
        output = inblock(x, 128, sn=False, name='block_0') # 16 x 16
        output = resblock_down(output, 128, sn=False, name='block_1') # 8 x 8
        output = resblock_down(output, 128, sn=False, downsample=False, name='block_2')
        output = resblock_down(output, 128, sn=False, downsample=False, name='block_3')

        # output layers
        output = tf.nn.relu(output)
        output = tf.reduce_sum(output, [1, 2])
        output = linear(output, 2*latent_dim, sn=False, name='linear')

        return output[:, :latent_dim], output[:, latent_dim:]


def celeba_generator(z, is_training=True):
    net_dim = 64
    use_sn = False
    update_collection = None
    with tf.variable_scope('Generator', reuse=tf.AUTO_REUSE):
        output = linear(z, 4 * 4 * 8 * net_dim, sn=use_sn, name='linear')
        output = batch_norm(output, is_training=is_training, name='bn_linear')
        output = tf.reshape(output, [-1, 4, 4, 8 * net_dim])

        # deconv-bn-relu
        for i in range(3):
            output = deconv2d(output, 2 ** (2 - i) * net_dim, sn=use_sn, name='deconv_' + str(i))
            output = batch_norm(output, is_training=is_training, name='bn_' + str(i))
            output = tf.nn.relu(output)
        # conv
        output = deconv2d(output, 3, sn=use_sn, name='deconv_out')
        output = tf.tanh(output)

        return output

def celeba_generator_resnet(z, is_training=False):
    with tf.variable_scope('Generator', reuse=tf.AUTO_REUSE):
        output = linear(z, 4 * 4 * 256, name='linear_0')
        output = tf.reshape(output, [-1, 4, 4, 256])
        output = resblock_up(output, 256, is_training=is_training, name='block_0')
        output = resblock_up(output, 256, is_training=is_training, name='block_1')
        output = resblock_up(output, 256, is_training=is_training, name='block_2')
        output = resblock_up(output, 256, is_training=is_training, name='block_3')

        # output layers
        output = batch_norm(output, is_training=is_training, name='g_bn')
        output = tf.nn.relu(output)
        output = conv2d(output, 3, 3, 1, name='conv_last')
        output = tf.tanh(output)

        return output


def celeba_discriminator(x, update_collection=None, is_training=False):
    net_dim = 64
    use_sn = True
    with tf.variable_scope('Discriminator', reuse=tf.AUTO_REUSE):
        # block 1
        x = conv2d(x, net_dim, 3, 1, sn=use_sn, update_collection=update_collection, name='conv0')
        x = lrelu(x)
        x = conv2d(x, net_dim, 4, 2, sn=use_sn, update_collection=update_collection, name='conv1')
        x = lrelu(x)
        # block 2
        x = conv2d(x, 2*net_dim, 3, 1, sn=use_sn, update_collection=update_collection, name='conv2')
        x = lrelu(x)
        x = conv2d(x, 2*net_dim, 4, 2, sn=use_sn, update_collection=update_collection, name='conv3')
        x = lrelu(x)
        # block 3
        x = conv2d(x, 4*net_dim, 3, 1, sn=use_sn, update_collection=update_collection, name='conv4')
        x = lrelu(x)
        x = conv2d(x, 4*net_dim, 4, 2, sn=use_sn, update_collection=update_collection, name='conv5')
        x = lrelu(x)
        # block 4
        x = conv2d(x, 4*net_dim, 3, 1, sn=use_sn, update_collection=update_collection, name='conv6')
        x = lrelu(x)
        x = conv2d(x, 4*net_dim, 4, 2, sn=use_sn, update_collection=update_collection, name='conv7')
        x = lrelu(x)
        # output
        x = conv2d(x, 8*net_dim, 3, 1, sn=use_sn, update_collection=update_collection, name='output')
        x = tf.reshape(x, [-1, 4*4*8*net_dim])
        x = linear(x, 1, sn=use_sn, update_collection=update_collection, name='linear')
        return tf.reshape(x, [-1])

def celeba_discriminator_resnet(x, update_collection=None, is_training=False):
    with tf.variable_scope('Discriminator', reuse=tf.AUTO_REUSE):
        output = inblock(x, 128, sn=True, update_collection=update_collection, name='block_0') # 16 x 16
        output = resblock_down(output, 128, sn=True, update_collection=update_collection, name='block_1') # 8 x 8
        output = resblock_down(output, 128, sn=True, update_collection=update_collection, downsample=False, name='block_2')
        output = resblock_down(output, 128, sn=True, update_collection=update_collection, downsample=False, name='block_3')
        output = resblock_down(output, 128, sn=True, update_collection=update_collection, downsample=False, name='block_4')

        # output layers
        output = tf.nn.relu(output)
        output = tf.reduce_sum(output, [1, 2])
        output = linear(output, 1, sn=True, update_collection=update_collection, name='linear')

        return tf.reshape(output, [-1])


def celeba_encoder(x, is_training=False, use_bn=False, net_dim=64, latent_dim=128):
    with tf.variable_scope('Encoder', reuse=tf.AUTO_REUSE):
        x = conv2d(x, net_dim, 3, 1, name='conv0')
        if use_bn:
            x = batch_norm(x, is_training=is_training, name='bn0')
        x = tf.nn.relu(x)

        x = conv2d(x, 2*net_dim, 4, 2, name='conv1')
        if use_bn:
            x = batch_norm(x, is_training=is_training, name='bn1')
        x = tf.nn.relu(x)

        x = conv2d(x, 4*net_dim, 4, 2, name='conv2')
        if use_bn:
            x = batch_norm(x, is_training=is_training, name='bn2')
        x = tf.nn.relu(x)

        x = conv2d(x, 8*net_dim, 4, 2, name='conv3')
        if use_bn:
            x = batch_norm(x, is_training=is_training, name='bn3')
        x = tf.nn.relu(x)

        x = conv2d(x, 8*net_dim, 4, 2, name='conv4')
        if use_bn:
            x = batch_norm(x, is_training=is_training, name='bn3')
        x = tf.nn.relu(x)

        x = tf.reshape(x, [-1, 4 * 4 * 8 * net_dim])
        x = linear(x, 2*latent_dim, name='linear')

        return x[:, :latent_dim], x[:, latent_dim:]


def celeba_encoder_resnet(x, is_training=False, use_bn=False, net_dim=64, latent_dim=128):
    with tf.variable_scope('Encoder', reuse=tf.AUTO_REUSE):
        output = inblock(x, 128, sn=False, name='block_0') # 16 x 16
        output = resblock_down(output, 128, sn=False, name='block_1') # 8 x 8
        output = resblock_down(output, 128, sn=False, downsample=False, name='block_2')
        output = resblock_down(output, 128, sn=False, downsample=False, name='block_3')
        output = resblock_down(output, 128, sn=False, downsample=False, name='block_4')

        # output layers
        output = tf.nn.relu(output)
        output = tf.reduce_sum(output, [1, 2])
        output = linear(output, 2*latent_dim, sn=False, name='linear')

        return output[:, :latent_dim], output[:, latent_dim:]


GENERATOR_DICT = {'mnist': [mnist_generator, mnist_generator],
                  'f-mnist': [mnist_generator, mnist_generator],
                  'cifar-10': [cifar10_generator, cifar10_generator_resnet],
                  'celeba': [celeba_generator, celeba_generator_resnet]}

DISCRIMINATOR_DICT = {'mnist': [mnist_discriminator, mnist_discriminator],
                      'f-mnist': [mnist_discriminator, mnist_discriminator],
                      'cifar-10': [cifar10_discriminator, cifar10_discriminator_resnet],
                      'celeba': [celeba_discriminator, celeba_discriminator_resnet]}

ENCODER_DICT = {'mnist': [mnist_encoder, mnist_encoder],
                'f-mnist': [mnist_encoder, mnist_encoder],
                'cifar-10': [cifar10_encoder, cifar10_encoder_resnet],
                'celeba': [celeba_encoder, celeba_encoder_resnet]
                }


def get_generator_fn(dataset_name, use_resblock=False):
    if use_resblock:
        return GENERATOR_DICT[dataset_name][1]
    else:
        return GENERATOR_DICT[dataset_name][0]


def get_discriminator_fn(dataset_name, use_resblock=False, use_label=False):
    if use_resblock:
        return DISCRIMINATOR_DICT[dataset_name][1]
    else:
        return DISCRIMINATOR_DICT[dataset_name][0]


def get_encoder_fn(dataset_name, use_resblock=False):
    if use_resblock:
        return ENCODER_DICT[dataset_name][1]
    else:
        return ENCODER_DICT[dataset_name][0]
