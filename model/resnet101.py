import tensorflow as tf


def conv2D(input, filters, kernel_size=3, stride=1, padding='SAME', d_rate=1):
    return tf.layers.conv2d(input, filters=filters, kernel_size=kernel_size,
                            padding=padding, dilation_rate=d_rate, strides=stride,
                            kernel_initializer=tf.variance_scaling_initializer())


def bn(input, is_training=True):
    return tf.layers.batch_normalization(input, training=is_training)


def conv_block(input, is_training, filters, d_rates, stride=1):
    x = bn(input, is_training)
    x = tf.nn.relu(x)
    x = conv2D(x, filters[0], kernel_size=1, d_rate=d_rates[0])

    x = bn(x, is_training)
    x = tf.nn.relu(x)
    x = conv2D(x, filters[1], kernel_size=3, stride=stride, d_rate=d_rates[1])

    x = bn(x, is_training)
    x_ = tf.nn.relu(x)
    x = conv2D(x_, filters[2], kernel_size=1, d_rate=d_rates[2])

    shortcut = tf.nn.relu(bn(input, is_training))
    shortcut = conv2D(shortcut, filters[2], kernel_size=1, stride=stride)
    x = tf.add(x, shortcut)
    return x


def identity_block(input, is_training, filters, d_rates):
    x = bn(input, is_training)
    x = tf.nn.relu(x)
    x = conv2D(x, filters[0], kernel_size=1, d_rate=d_rates[0])

    x = bn(x, is_training)
    x = tf.nn.relu(x)
    x = conv2D(x, filters[1], kernel_size=3, d_rate=d_rates[1])

    x = bn(x, is_training)
    x = tf.nn.relu(x)
    x = conv2D(x, filters[2], kernel_size=1, d_rate=d_rates[2])

    x = tf.add(x, input)
    return x


def resnet101(input, is_training=True):
    conv_1 = conv2D(input, 64, stride=2)
    conv_1 = bn(conv_1, is_training)
    conv_1 = tf.nn.relu(conv_1)
    conv_2 = conv2D(conv_1, 64)
    conv_2 = bn(conv_2, is_training)
    conv_2 = tf.nn.relu(conv_2)
    conv_3 = conv2D(conv_2, 128)
    conv_3 = bn(conv_3, is_training)
    conv_3 = tf.nn.relu(conv_3)
    pool1 = tf.layers.max_pooling2d(conv_3, 2, 2)

    x = conv_block(pool1, is_training, filters=[64, 64, 256], stride=1, d_rates=[1, 1, 1])
    x = identity_block(x, is_training, filters=[64, 64, 256], d_rates=[1, 1, 1])
    x1 = identity_block(x, is_training, filters=[64, 64, 256], d_rates=[1, 1, 1])

    x = conv_block(x1, is_training, filters=[128, 128, 512], stride=2, d_rates=[1, 1, 1])
    x = identity_block(x, is_training, filters=[128, 128, 512], d_rates=[1, 1, 1])
    x = identity_block(x, is_training, filters=[128, 128, 512], d_rates=[1, 1, 1])
    x2 = identity_block(x, is_training, filters=[128, 128, 512], d_rates=[1, 1, 1])

    x = conv_block(x2, is_training, filters=[256, 256, 1024], stride=1, d_rates=[1, 2, 1])
    x = identity_block(x, is_training, filters=[256, 256, 1024], d_rates=[1, 2, 1])
    x = identity_block(x, is_training, filters=[256, 256, 1024], d_rates=[1, 2, 1])
    x = identity_block(x, is_training, filters=[256, 256, 1024], d_rates=[1, 2, 1])
    x = identity_block(x, is_training, filters=[256, 256, 1024], d_rates=[1, 2, 1])

    x = identity_block(x, is_training, filters=[256, 256, 1024], d_rates=[1, 2, 1])
    x = identity_block(x, is_training, filters=[256, 256, 1024], d_rates=[1, 2, 1])
    x = identity_block(x, is_training, filters=[256, 256, 1024], d_rates=[1, 2, 1])
    x = identity_block(x, is_training, filters=[256, 256, 1024], d_rates=[1, 2, 1])

    x = identity_block(x, is_training, filters=[256, 256, 1024], d_rates=[1, 2, 1])
    x = identity_block(x, is_training, filters=[256, 256, 1024], d_rates=[1, 2, 1])
    x = identity_block(x, is_training, filters=[256, 256, 1024], d_rates=[1, 2, 1])
    x = identity_block(x, is_training, filters=[256, 256, 1024], d_rates=[1, 2, 1])

    x = identity_block(x, is_training, filters=[256, 256, 1024], d_rates=[1, 2, 1])
    x = identity_block(x, is_training, filters=[256, 256, 1024], d_rates=[1, 2, 1])
    x = identity_block(x, is_training, filters=[256, 256, 1024], d_rates=[1, 2, 1])
    x = identity_block(x, is_training, filters=[256, 256, 1024], d_rates=[1, 2, 1])

    x = identity_block(x, is_training, filters=[256, 256, 1024], d_rates=[1, 2, 1])
    x = identity_block(x, is_training, filters=[256, 256, 1024], d_rates=[1, 2, 1])
    x = identity_block(x, is_training, filters=[256, 256, 1024], d_rates=[1, 2, 1])
    x = identity_block(x, is_training, filters=[256, 256, 1024], d_rates=[1, 2, 1])
    x = identity_block(x, is_training, filters=[256, 256, 1024], d_rates=[1, 2, 1])

    x3 = identity_block(x, is_training, filters=[256, 256, 1024], d_rates=[1, 2, 1])

    x = conv_block(x3, is_training, filters=[512, 512, 2048], d_rates=[1, 4, 1])
    x = identity_block(x, is_training, filters=[512, 512, 2048], d_rates=[1, 4, 1])
    bottom = identity_block(x, is_training, filters=[512, 512, 2048], d_rates=[1, 4, 1])

    x = conv2D(bottom, 512)
    x = tf.nn.relu(bn(x, is_training))
    x = conv2D(x, 1, 1)
    final = tf.image.resize_bilinear(x, input.shape[1:3])

    return final

