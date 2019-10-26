import tensorflow as tf


def conv2D(input, filters, kernel_size=3, stride=1, padding='SAME', d_rate=1):
    return tf.layers.conv2d(input, filters=filters, kernel_size=kernel_size,
                            padding=padding, dilation_rate=d_rate, strides=stride,
                            kernel_initializer=tf.variance_scaling_initializer())


def bn(input, is_training=True):
    return tf.layers.batch_normalization(inputs=input,training=is_training,
                                      center=True,scale=True,fused=True)


def unet(input,is_training=True):
    conv1 = conv2D(input, 32)
    bn1 = tf.nn.relu(bn(conv1, is_training))
    conv1_1 = conv2D(bn1, 32)
    bn1_1 = tf.nn.relu(bn(conv1_1, is_training))
    pool1 = tf.layers.max_pooling2d(bn1_1, 2, 2)

    conv2 = conv2D(pool1, 64)
    bn2 = tf.nn.relu(bn(conv2, is_training))
    conv2_1 = conv2D(bn2, 64)
    bn2_1 = tf.nn.relu(bn(conv2_1, is_training))
    pool2 = tf.layers.max_pooling2d(bn2_1, 2, 2)

    conv3 = conv2D(pool2, 128)
    bn3 = tf.nn.relu(bn(conv3, is_training))
    conv3_1 = conv2D(bn3, 128)
    bn3_1 = tf.nn.relu(bn(conv3_1, is_training))
    pool3 = tf.layers.max_pooling2d(bn3_1, 2, 2)

    conv4 = conv2D(pool3, 256)
    bn4 = tf.nn.relu(bn(conv4, is_training))
    conv4_1 = conv2D(bn4, 256)
    bn4_1 = tf.nn.relu(bn(conv4_1, is_training))
    pool4 = tf.layers.max_pooling2d(bn4_1, 2, 2)

    conv5 = conv2D(pool4, 512)
    bn5 = tf.nn.relu(bn(conv5, is_training))
    conv5_1 = conv2D(bn5, 512)
    bn5_1 = tf.nn.relu(bn(conv5_1, is_training))

    up1 = tf.image.resize_images(bn5_1, tf.shape(bn5_1)[1:3] * 2)
    up1 = conv2D(up1, 256)
    up1 = tf.concat([up1, bn4_1], axis=3)
    up1 = conv2D(up1, 256)
    up1 = tf.nn.relu(bn(up1, is_training))
    up1 = conv2D(up1, 256)
    up1 = tf.nn.relu(bn(up1, is_training))

    up2 = tf.image.resize_images(up1, tf.shape(up1)[1:3] * 2)
    up2 = conv2D(up2, 128)
    up2 = tf.concat([up2, bn3_1], axis=3)
    up2 = conv2D(up2, 128)
    up2 = tf.nn.relu(bn(up2, is_training))
    up2 = conv2D(up2, 128)
    up2 = tf.nn.relu(bn(up2, is_training))

    up3 = tf.image.resize_images(up2, tf.shape(up2)[1:3] * 2)
    up3 = conv2D(up3, 64)
    up3 = tf.concat([up3, bn2_1], axis=3)
    up3 = conv2D(up3, 64)
    up3 = tf.nn.relu(bn(up3, is_training))
    up3 = conv2D(up3, 64)
    up3 = tf.nn.relu(bn(up3, is_training))

    up4 = tf.image.resize_images(up3, tf.shape(up3)[1:3] * 2)
    up4 = conv2D(up4, 32)
    up4 = tf.concat([up4, bn1_1], axis=3)
    up4 = conv2D(up4, 32)
    up4 = tf.nn.relu(bn(up4, is_training))
    up4 = conv2D(up4, 32)
    up4 = tf.nn.relu(bn(up4, is_training))

    final = conv2D(up4, 1, 1)
    return final

