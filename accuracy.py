import numpy as np
import scipy.misc
import glob


data1 = sorted(glob.glob(r'/media/lc/vge_lc/DL_DATE_BUILDING/WHU/cropped image tiles and raster labels/test/gt/*.png'))
# data1 = sorted(glob.glob(r'/home/lc/Jupyter_projects/resatt/Urban 3D Challenge Data/d_test/gt/*.png'))
# data1=sorted(glob.glob('/media/lc/vge_lc/spacenet/shanghai_vegas_test_result/test_label/*.png'))
# data1=sorted(glob.glob('/media/lc/vge_lc/urban3/uba512_test/test_gt/*.png'))
data2 = sorted(glob.glob('./test_result_temp/*.png'))


def iou(predict, label):
    Intersect = []
    Union = []
    for i in range(2):
        Ii = np.sum(np.logical_and(predict == i, label == i))
        Ui = np.sum(predict == i) + np.sum(label == i) - np.sum(np.logical_and(predict == i, label == i))
        Intersect.append(Ii)
        Union.append(Ui)
    return Intersect, Union


def f_score(predict, label):
    tp1 = []
    fp1 = []
    tn1 = []
    fn1 = []
    for i in range(1, 2):
        tp = np.sum(np.logical_and(predict == i, label == i))
        fp = np.sum(np.logical_and(predict == i, label != i))
        tn = np.sum(np.logical_and(predict != i, label != i))
        fn = np.sum(np.logical_and(predict != i, label == i))
        tp1.append(tp)
        fp1.append(fp)
        tn1.append(tn)
        fn1.append(fn)
    return tp1, fp1, tn1, fn1


def cal_iou():
    l1 = []
    l2 = []
    l3 = []
    l4 = []
    TP = []
    FP = []
    TN = []
    FN = []
    for i in range(len(data1)):
        label = scipy.misc.imread(data1[i])
        predict = scipy.misc.imread(data2[i])
        predict = predict // 255
        ap = np.sum(label == predict)
        total = np.sum(label != 2)
        Inter, Uni = iou(predict, label)
        tp1, fp1, tn1, fn1 = f_score(predict, label)
        TP.append(tp1)
        FP.append(fp1)
        TN.append(tn1)
        FN.append(fn1)

        l1.append(Inter)
        l2.append(Uni)
        l3.append(ap)
        l4.append(total)

    a = np.sum(l1, axis=0)
    b = np.sum(l2, axis=0)
    IoU = a * 1.0 / b
    print('iou:{}'.format(IoU))
    mean_iu = np.sum(IoU[:2]) / 2
    print('mean_iu:{}'.format(mean_iu))
    precision = np.sum(TP, axis=0) / (np.sum(TP, axis=0) + np.sum(FP, axis=0))
    print('--precision:{}'.format(precision))
    recall = np.sum(TP, axis=0) / (np.sum(TP, axis=0) + np.sum(FN, axis=0))
    print('--recall:{}'.format(recall))
    F_score = 2 * (precision * recall) / (precision + recall)
    print('F_score:{}'.format(F_score))
    mean_ap = np.sum(l3) * 1.0 / np.sum(l4)
    print('mean_ap:{}'.format(mean_ap))
    return IoU, mean_iu, mean_ap


c, d, mean_ap = cal_iou()

