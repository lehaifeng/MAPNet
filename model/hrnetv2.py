import tensorflow as tf
from keras.layers import UpSampling2D


def conv2d(input,filters,kernel_size=3,stride=1,padding='SAME'):
    return tf.layers.conv2d(input,filters=filters,kernel_size=kernel_size,
                            padding=padding,strides=stride,use_bias=False,
                            kernel_initializer=tf.variance_scaling_initializer())


def bn(input,is_training=True):
    return tf.layers.batch_normalization(input,momentum=0.1,epsilon=1e-5,training=is_training)


def Bottleneck(x, size,is_training,downsampe=False):
    residual = x

    out = bn(x, is_training)
    out = tf.nn.relu(out)
    out = conv2d(out, size, 1, padding='VALID')
    out = bn(out, is_training)
    out = tf.nn.relu(out)
    out = conv2d(out, size, 3)
    out = bn(out, is_training)
    out = tf.nn.relu(out)
    out = conv2d(out, size * 4, 1, padding='VALID')

    if downsampe:
        residual = bn(x, is_training)
        out = tf.nn.relu(out)
        residual = conv2d(residual, size * 4, 1, padding='VALID')
    out = tf.add(out,residual)
    return out


def BasicBlock(x, size,is_training, downsampe=False):
    residual = x
    out = bn(x, is_training)
    out = tf.nn.relu(out)
    out = conv2d(out, size, 3)
    out = bn(out, is_training)
    out = tf.nn.relu(out)
    out = conv2d(out, size, 3)
    if downsampe:
        residual = bn(x, is_training)
        residual = tf.nn.relu(residual)
        residual = conv2d(residual, size, 1, padding='VALID')

    out = tf.add(out, residual)
    return out


def layer1(x,is_training):
    x = Bottleneck(x, 64,is_training,downsampe=True)
    x = Bottleneck(x, 64,is_training)
    x = Bottleneck(x, 64,is_training)
    x = Bottleneck(x, 64,is_training)
    return x


def transition_layer(x, in_channels, out_channels,is_training):
    num_in = len(in_channels)
    num_out = len(out_channels)
    out = []
    for i in range(num_out):
        if i < num_in:
            if in_channels[i] != out_channels[i]:
                residual = bn(x[i], is_training)
                residual = tf.nn.relu(residual)
                residual = conv2d(residual, out_channels[i], 3)
                out.append(residual)
            else:
                out.append(x[i])
        else:
            residual = bn(x[-1], is_training)
            residual = tf.nn.relu(residual)
            residual = conv2d(residual, out_channels[i], 3, stride=2)
            out.append(residual)
    return out


def branches(x, block_num, channels,is_training):
    out = []
    for i in range(len(channels)):
        residual = x[i]
        for j in range(block_num):
            residual = BasicBlock(residual, channels[i],is_training)
        out.append(residual)
    return out


def fuse_layers(x, channels,is_training, multi_scale_output=True):
    out = []
    for i in range(len(channels) if multi_scale_output else 1):
        residual = x[i]
        for j in range(len(channels)):
            if j > i:
                y = bn(x[j], is_training)
                y = tf.nn.relu(y)
                y = conv2d(y, channels[i], 1, padding='VALID')
                y = UpSampling2D(size=2 ** (j - i))(y)
                residual = tf.add(residual,y)
            elif j < i:
                y = x[j]
                for k in range(i - j):
                    if k == i - j - 1:
                        y = bn(y, is_training)
                        y = tf.nn.relu(y)
                        y = conv2d(y, channels[i], 3, stride=2)
                    else:
                        y = bn(y, is_training)
                        y = tf.nn.relu(y)
                        y = conv2d(y, channels[j], 3, stride=2)

                residual = tf.add(residual,y)
        out.append(residual)
    return out


def HighResolutionModule(x, channels,is_training, multi_scale_output=True):
    residual = branches(x, 4, channels,is_training)
    out = fuse_layers(residual, channels,is_training,multi_scale_output=multi_scale_output)
    return out


def stage(x, num_modules, channels, is_training,multi_scale_output=True):
    out = x
    for i in range(num_modules):
        if i == num_modules - 1 and multi_scale_output == False:
            out = HighResolutionModule(out, channels,is_training, multi_scale_output=False)
        else:
            out = HighResolutionModule(out, channels,is_training)
    return out


def hrnetv2(input,is_training=True):
    channels_2 = [32, 64]
    channels_3 = [32, 64, 128]
    channels_4 = [32, 64, 128, 256]
    num_modules_2 = 1
    num_modules_3 = 4
    num_modules_4 = 3

    x = conv2d(input, 64, 3, stride=2)
    x = bn(x,is_training)
    x = tf.nn.relu(x)
    x = conv2d(x, 64, 3, stride=2)
    x = bn(x,is_training)
    x = tf.nn.relu(x)

    la1 = layer1(x,is_training)
    tr1 = transition_layer([la1], [256], channels_2,is_training)

    st2 = stage(tr1, num_modules_2, channels_2,is_training)
    tr2 = transition_layer(st2, channels_2, channels_3,is_training)

    st3 = stage(tr2, num_modules_3, channels_3,is_training)
    tr3 = transition_layer(st3, channels_3, channels_4,is_training)
    st4 = stage(tr3, num_modules_4, channels_4,is_training, multi_scale_output=False)

    up1=tf.image.resize_bilinear(st4[0],size=(st4[0].shape[1]*2,st4[0].shape[2]*2))
    up1 = bn(up1, is_training)
    up1 = tf.nn.relu(up1)
    up1 = conv2d(up1, 32, 3)

    up2 = tf.image.resize_bilinear(up1, size=(up1.shape[1]*2, up1.shape[2]*2))
    up2 = bn(up2, is_training)
    up2 = tf.nn.relu(up2)
    up2 = conv2d(up2, 32, 3)

    up2 = bn(up2, is_training)
    up2 = tf.nn.relu(up2)
    final = conv2d(up2, 1, 1, padding='VALID')
    return final

