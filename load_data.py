
import numpy as np
import glob
import scipy
import random
import cv2


def load_batch(x, y):
    x1 = []
    y1 = []
    for i in range(len(x)):
        img = scipy.misc.imread(x[i])
        lab = scipy.misc.imread(y[i])
        img, lab = data_augmentation(img, lab)
        lab = lab.reshape(512, 512, 1)
        x1.append(img / 255.0)
        y1.append(lab)
    y1 = np.array(y1).astype(np.float32)
    return x1, y1


def prepare_data():
    # whu 512*512 4736
    # img = np.array(sorted(
    #     glob.glob(r'/media/lc/vge_lc/DL_DATE_BUILDING/WHU/cropped image tiles and raster labels/train/image/*.png')))
    # label = np.array(sorted(
    #     glob.glob(r'/media/lc/vge_lc/DL_DATE_BUILDING/WHU/cropped image tiles and raster labels/train/gt/*.png')))
    #
    # test_img = np.array(sorted(
    #     glob.glob(r'/media/lc/vge_lc/DL_DATE_BUILDING/WHU/cropped image tiles and raster labels/test/image/*.png')))
    # test_label = np.array(sorted(
    #     glob.glob(r'/media/lc/vge_lc/DL_DATE_BUILDING/WHU/cropped image tiles and raster labels/test/gt/*.png')))

    img = np.array(sorted(glob.glob(r'./dataset/train/img/*.png')))
    label = np.array(sorted(glob.glob(r'./dataset/train/lab/*.png')))
    test_img = np.array(sorted(glob.glob(r'./dataset/test/img/*.png')))
    test_label = np.array(sorted(glob.glob(r'./dataset/test/lab/*.png')))


    # img = np.array(sorted(glob.glob(r'/media/lc/vge_lc/spacenet/train_rgb_image/*.png')))
    # label = np.array(sorted(glob.glob(r'/media/lc/vge_lc/spacenet/train_label_image/*.png')))
    # test_img=sorted(glob.glob(r'/media/lc/vge_lc/spacenet/shanghai_vegas_test_result/test_image/*.png'))
    # test_label=sorted(glob.glob('/media/lc/vge_lc/spacenet/shanghai_vegas_test_result/test_label/*.png'))

    # img = np.array(sorted(glob.glob(r'/home/lc/Jupyter_projects/resatt/Urban 3D Challenge Data/d_train/img/*.png')))
    # label = np.array(sorted(glob.glob(r'/home/lc/Jupyter_projects/resatt/Urban 3D Challenge Data/d_train/gt/*.png')))
    # test_img = np.array(sorted(glob.glob(r'/home/lc/Jupyter_projects/resatt/Urban 3D Challenge Data/d_test/img/*.png')))
    # test_label = np.array(sorted(glob.glob(r'/home/lc/Jupyter_projects/resatt/Urban 3D Challenge Data/d_test/gt/*.png')))
    return img, label, test_img, test_label


def data_augmentation(image, label):
    # Data augmentation
    if random.randint(0, 1):
        image = np.fliplr(image)
        label = np.fliplr(label)
    if random.randint(0, 1):
        image = np.flipud(image)
        label = np.flipud(label)

    if random.randint(0,1):
        angle = random.randint(0, 3)*90
        if angle!=0:
            M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1.0)
            image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST)
            label = cv2.warpAffine(label, M, (label.shape[1], label.shape[0]), flags=cv2.INTER_NEAREST)

    return image, label

