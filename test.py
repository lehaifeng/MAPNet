import scipy
import tensorflow as tf
import os
import numpy as np
import glob

from model. mapnet import mapnet
# from model.hrnetv2 import hrnetv2
# from model.pspnet import pspnet
# from model.unet import unet
# from model.resnet101 import resnet101

batch_size = 1
img = tf.placeholder(tf.float32, [batch_size, 512, 512, 3])

# WHU
test_img = sorted(
    glob.glob(r'/media/lc/vge_lc/DL_DATE_BUILDING/WHU/cropped image tiles and raster labels/test/image/*.png'))
# SpaceNet
# test_img=sorted(glob.glob(r'/media/lc/vge_lc/spacenet/shanghai_vegas_test_result/test_image/*.png'))
# Urban
# test_img = np.array(sorted(glob.glob(r'/home/lc/Jupyter_projects/resatt/Urban 3D Challenge Data/d_test/img/*.png')))

pred = mapnet(img, is_training=False)
pred = tf.nn.sigmoid(pred)
saver = tf.train.Saver(tf.global_variables())


def save():
    tf.global_variables_initializer().run()
    checkpoint_dir = './checkpoint/'
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))

    for j in range(0, len(test_img)):
        x_batch = test_img[j]
        i = x_batch.split('/')[-1]
        x_batch = scipy.misc.imread(x_batch) / 255.0
        x_batch = np.expand_dims(x_batch, axis=0)
        feed_dict = {img: x_batch

                     }
        predict = sess.run(pred, feed_dict=feed_dict)
        predict[predict < 0.5] = 0
        predict[predict >= 0.5] = 1
        result = np.squeeze(predict)
        i = i.split('.')[0]
        scipy.misc.imsave('./test_result_temp/{}.png'.format(i), result)


with tf.Session() as sess:
    save()

