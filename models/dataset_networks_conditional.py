from tflib.layers import *


def mnist_generator_conditional(z, target_class, num_classes, is_training=True):
    net_dim = 64
    use_sn = False
    update_collection = None
    with tf.variable_scope('Generator', reuse=tf.AUTO_REUSE):
        bn_linear = ConditionalBatchNorm(num_classes, name='cbn_linear')
        bn_0 = ConditionalBatchNorm(num_classes, name='cbn_0')
        bn_1 = ConditionalBatchNorm(num_classes, name='cbn_1')

        output = linear(z, 4*4*4*net_dim, sn=use_sn, name='linear')
        output = bn_linear(output, target_class, is_training=is_training)
        output = tf.nn.relu(output)
        output = tf.reshape(output, [-1, 4, 4, 4*net_dim])

        # deconv-bn-relu
        output = deconv2d(output, 2*net_dim, 5, 2, sn=use_sn, name='deconv_0')
        output = bn_0(output, target_class, is_training=is_training)
        output = tf.nn.relu(output)

        output = output[:, :7, :7, :]

        output = deconv2d(output, net_dim, 5, 2, sn=use_sn, name='deconv_1')
        output = bn_1(output, target_class, is_training=is_training)
        output = tf.nn.relu(output)

        output = deconv2d(output, 1, 5, 2, sn=use_sn, name='deconv_2')
        output = tf.sigmoid(output)

        return output


def mnist_discriminator_conditional(x, target_class, num_classes, update_collection=None, is_training=False):
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
        h = tf.reshape(x, [-1, 4 * 4 * 4 * net_dim])
        output = linear(h, 1, sn=use_sn, update_collection=update_collection, name='linear')
        h_labels = embedding(target_class, number_classes=num_classes, embedding_size=64*net_dim, update_collection=update_collection)

        output += tf.reduce_sum(h * h_labels, axis=1, keepdims=True)

        return tf.reshape(output, [-1])


def cifar10_generator_conditional(z, target_class, num_classes, is_training=True):
    net_dim = 64
    use_sn = False
    update_collection = None
    with tf.variable_scope('Generator', reuse=tf.AUTO_REUSE):
        bn_linear = ConditionalBatchNorm(num_classes, name='cbn_linear')
        bn_list = [ConditionalBatchNorm(num_classes, name='cbn_0'),
                   ConditionalBatchNorm(num_classes, name='cbn_1'),
                   ConditionalBatchNorm(num_classes, name='cbn_2')]

        output = linear(z, 4 * 4 * 8 * net_dim, sn=use_sn, name='linear')
        output = bn_linear(output, target_class, is_training=is_training)
        output = tf.nn.relu(output)
        output = tf.reshape(output, [-1, 4, 4, 8 * net_dim])

        # deconv-bn-relu
        for i in range(3):
            output = deconv2d(output, 2 ** (2 - i) * net_dim, sn=use_sn, name='deconv_' + str(i))
            output = bn_list[i](output, target_class, is_training=is_training)
            output = tf.nn.relu(output)
        # conv
        output = conv2d(output, 3, sn=use_sn, name='conv3')
        output = tf.tanh(output)

        return output


def cifar10_discriminator_conditional(x, target_class, num_classes, update_collection=None, is_training=False):
    net_dim = 64
    use_sn = True
    with tf.variable_scope('Discriminator', reuse=tf.AUTO_REUSE):
        # block 1
        x = conv2d(x, net_dim, 3, 1, sn=use_sn, update_collection=update_collection, name='conv0')
        x = lrelu(x)
        x = conv2d(x, net_dim, 4, 2, sn=use_sn, update_collection=update_collection, name='conv1')
        x = lrelu(x)
        # block 2
        x = conv2d(x, 2 * net_dim, 3, 1, sn=use_sn, update_collection=update_collection, name='conv2')
        x = lrelu(x)
        x = conv2d(x, 2 * net_dim, 4, 2, sn=use_sn, update_collection=update_collection, name='conv3')
        x = lrelu(x)
        # block 3
        x = conv2d(x, 4 * net_dim, 3, 1, sn=use_sn, update_collection=update_collection, name='conv4')
        x = lrelu(x)
        x = conv2d(x, 4 * net_dim, 4, 2, sn=use_sn, update_collection=update_collection, name='conv5')
        x = lrelu(x)
        # output
        x = conv2d(x, 8 * net_dim, 3, 1, sn=use_sn, update_collection=update_collection, name='conv6')
        h = tf.reshape(x, [-1, 4 * 4 * 8 * net_dim])
        output = linear(h, 1, sn=use_sn, update_collection=update_collection, name='linear')
        h_labels = embedding(target_class, number_classes=num_classes, embedding_size=128 * net_dim,
                             update_collection=update_collection)

        output += tf.reduce_sum(h * h_labels, axis=1, keepdims=True)

        return tf.reshape(output, [-1])


GENERATOR_DICT = {'mnist': [mnist_generator_conditional],
                  'f-mnist': [mnist_generator_conditional],
                  'cifar-10': [cifar10_generator_conditional],
                  'celeba': [None]}

DISCRIMINATOR_DICT = {'mnist': [mnist_discriminator_conditional],
                      'f-mnist': [mnist_discriminator_conditional],
                      'cifar-10': [cifar10_discriminator_conditional],
                      'celeba': [None]}


def get_generator_fn(dataset_name):
    return GENERATOR_DICT[dataset_name][0]


def get_discriminator_fn(dataset_name):
    return DISCRIMINATOR_DICT[dataset_name][0]