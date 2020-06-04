import os
import time
import torch.nn as nn
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import torch.optim as optim
import torchvision.models as models
from tools.common_tools import set_seed
from torch.utils.tensorboard import SummaryWriter
from tools.my_dataset import SkyDataset
from tools.unet import UNet
from tools.LovaszLoss import lovasz_hinge
from torch.utils.data import SubsetRandomSampler
import cv2
import func
import copy

def find_sky_rect(mask):
    sz = mask.shape
    rows = sz[0]
    cols = sz[1]

    col = 0
    c1 = -1
    c2 = -1
    r1 = -1
    r2 = -1

    while col <= cols - 1:
        column = mask[:, col]
        flag_c = column.any()
        if c2 == -1 and flag_c:
            c2 = col
        elif c2 != -1 and flag_c:  # 已有起始位置，当前列不为空
            c1 = col
        col = col + 1

    row = 0
    while row <= rows - 1:
        row_vec = mask[row, :]
        flag_r = row_vec.any()
        if r2 == -1 and flag_r:
            r2 = row
        elif r2 != -1 and flag_r:
            r1 = row
        row = row + 1

    return (r1, c1, r2, c2)

def replace_sky(img, mask_bw, sky):
    r1, c1, r2, c2 = func.find_sky_rect(mask_bw)

    height = r1 - r2 + 1
    width = c1 - c2 + 1

    sky_resize = cv2.resize(sky, (width, height))

    I_rep = img.copy()
    sz = img.shape

    for i in range(sz[0]):
        for j in range(sz[1]):
            if (mask_bw[i, j].any()):
                I_rep[i, j, :] = sky_resize[i - r2, j - c2, :]
    return I_rep

def guideFilter(I, p, mask_edge, winSize, eps):  # input p,giude I

    I = I / 255.0
    p = p / 255.0
    mask_edge = mask_edge / 255.0
    # I的均值平滑
    mean_I = cv2.blur(I, winSize)

    # p的均值平滑
    mean_p = cv2.blur(p, winSize)

    # I*I和I*p的均值平滑
    mean_II = cv2.blur(I * I, winSize)

    mean_Ip = cv2.blur(I * p, winSize)

    # 方差
    var_I = mean_II - mean_I * mean_I  # 方差公式

    # 协方差
    cov_Ip = mean_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    # 对a、b进行均值平滑
    mean_a = cv2.blur(a, winSize)
    mean_b = cv2.blur(b, winSize)

    q = p.copy()
    sz = mask_edge.shape
    # edge=mask_edge.copy()
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    edge = cv2.dilate(mask_edge, kernel)

    # for i in range(sz[0]):
    # 	for j in range(sz[1]):
    # 		if (mask_edge[i,j].any()):
    # 			if (i+1<sz[0]):
    # 				edge[i+1,j]=1
    # 			if (i+2<sz[0]):
    # 				edge[i+2,j]=1
    # 			if (j-1>-1):
    # 				edge[i,j-1]=1
    # 			if (j+1<sz[1]):
    # 				edge[i,j+1]=1
    # 			if (j-2>-1):
    # 				edge[i,j-2]=1
    # 			if (j+2<sz[1]):
    # 				edge[i,j+2]=1
    # edge[i+2,j]=1
    # edge[i+3,j]=1

    q[edge == 1] = mean_a[edge == 1] * I[edge == 1] + mean_b[edge == 1]
    # for i in range(sz[0]):
    # 	for j in range(sz[1]):
    # 		if (edge[i,j].any()):
    # 			q[i,j] = mean_a[i,j]*I[i,j] + mean_b[i,j]

    # q = mean_a*I + mean_b

    q = q * 255
    q[q > 255] = 255
    q = np.round(q)
    q = q.astype(np.uint8)
    return q

def color_transfer(source, mask_bw, target, mode):
    """
    Transfers the color distribution from the source to the target
    image using the mean and standard deviations of the L*a*b*
    color space.

    This implementation is (loosely) based on to the "Color Transfer
    between Images" paper by Reinhard et al., 2001.

    Parameters:
    -------
    source: NumPy array
        OpenCV image in BGR color space (the source image)
    target: NumPy array
        OpenCV image in BGR color space (the target image)

    Returns:
    -------
    transfer: NumPy array
        OpenCV image (w, h, 3) NumPy array (uint8)
    """

    # convert the images from the RGB to L*ab* color space, being
    # sure to utilizing the floating point data type (note: OpenCV
    # expects floats to be 32-bit, so use that instead of 64-bit)

    source = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype("float32")

    (l, a, b) = cv2.split(target)
    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
    if mode:
        index = np.where(mask_bw == 0)
        index = np.array(index, dtype=int)
        sz = index.shape
        # fl=[]
        # fa=[]
        # fb=[]
        fl = target[:, :, 0][mask_bw == 0]
        fa = target[:, :, 1][mask_bw == 0]
        fb = target[:, :, 2][mask_bw == 0]

        # for i in range(sz[1]):
        # 	x=index[0,i]
        # 	y=index[1,i]
        # 	fl.append(target[x,y,0])
        # 	fa.append(target[x,y,1])
        # 	fb.append(target[x,y,2])

        (lMeanTar_1, lStdTar_1) = (np.mean(fl), np.std(fl, ddof=1))
        (aMeanTar_1, aStdTar_1) = (np.mean(fa), np.std(fa, ddof=1))
        (bMeanTar_1, bStdTar_1) = (np.mean(fb), np.std(fb, ddof=1))
        (lMeanTar_2, lStdTar_2, aMeanTar_2, aStdTar_2, bMeanTar_2, bStdTar_2) = image_stats(target)
        # (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)

        l[mask_bw == 0] -= lMeanTar_2
        a[mask_bw == 0] -= aMeanTar_2
        b[mask_bw == 0] -= bMeanTar_2

        l[mask_bw == 0] *= (lStdSrc / lStdTar_2)
        a[mask_bw == 0] *= (lStdSrc / lStdTar_2)
        b[mask_bw == 0] *= (lStdSrc / lStdTar_2)

        l[mask_bw == 0] += lMeanSrc
        a[mask_bw == 0] += aMeanSrc
        b[mask_bw == 0] += bMeanSrc

    # mask_size=mask_bw.shape
    # for i in range(mask_size[0]):
    # 	for j in range(mask_size[1]):
    # 		if mask_bw[i,j]==0:

    # 			l[i,j] -= lMeanTar_2
    # 			a[i,j] -= aMeanTar_2
    # 			b[i,j] -= bMeanTar_2
    # 			# scale by the standard deviations
    # 			l[i,j] = (lStdSrc / lStdTar_2) * l[i,j]
    # 			a[i,j] = (lStdSrc / lStdTar_2) * a[i,j]
    # 			b[i,j] = (lStdSrc / lStdTar_2) * b[i,j]
    # 			# add in the source mean
    # 			l[i,j] += lMeanSrc
    # 			a[i,j] += aMeanSrc
    # 			b[i,j] += bMeanSrc
    # else:
    # 	# (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)
    # 	# (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)

    # 	l[i,j] -= lMeanTar_2
    # 	a[i,j] -= aMeanTar_2
    # 	b[i,j] -= bMeanTar_2

    # 	# scale by the standard deviations
    # 	l[i,j] = (lStdTar_2 / lStdSrc) * l[i,j]
    # 	a[i,j] = (aStdTar_2 / aStdSrc) * a[i,j]
    # 	b[i,j] = (bStdTar_2 / bStdSrc) * b[i,j]

    # # 	# add in the source mean
    # # 	# l[i,j] += lMeanSrc
    # # 	# a[i,j] += aMeanSrc
    # # 	# b[i,j] += bMeanSrc
    else:
        (lMeanTar_2, lStdTar_2, aMeanTar_2, aStdTar_2, bMeanTar_2, bStdTar_2) = image_stats(target)
        l -= lMeanTar_2
        a -= aMeanTar_2
        b -= bMeanTar_2

        # scale by the standard deviations
        # l = (lStdTar_2 / lStdSrc) * l
        # a = (aStdTar_2 / aStdSrc) * a
        # b = (bStdTar_2 / bStdSrc) * b
        l = (lStdSrc / lStdTar_2) * l
        a = (aStdSrc / aStdTar_2) * a
        b = (bStdSrc / bStdTar_2) * b
        l += lMeanSrc
        a += aMeanSrc
        b += bMeanSrc

    # l += lMeanSrc
    # a += aMeanSrc
    # b += bMeanSrc

    # clip the pixel intensities to [0, 255] if they fall outside
    # this range
    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)

    # merge the channels together and convert back to the RGB color
    # space, being sure to utilize the 8-bit unsigned integer data
    # type
    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2RGB)

    # return the color transferred image
    return transfer

def image_stats(image):
    """
    Parameters:
    -------
    image: NumPy array
        OpenCV image in L*a*b* color space

    Returns:
    -------
    Tuple of mean and standard deviations for the L*, a*, and b*
    channels, respectively
    """
    # compute the mean and standard deviation of each channel
    (l, a, b) = cv2.split(image)
    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())

    # return the color statistics
    return (lMean, lStd, aMean, aStd, bMean, bStd)

def compute_dice(y_pred, y_true):
    """
    :param y_pred: 4-d tensor, value = [0,1]
    :param y_true: 4-d tensor, value = [0,1]
    :return: Dice index 2*TP/(2*TP+FP+FN)=2TP/(pred_P+true_P)
    """
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    y_pred, y_true = np.round(y_pred).astype(int), np.round(y_true).astype(int)
    return np.sum(y_pred[y_true == 1]) * 2.0 / (np.sum(y_pred) + np.sum(y_true))

##########################################################
def video_infer(img_pil):
    checkpoint_load = 'd:/MyLearning/DIP/Final_Project/Unet/test2_lovasz_1e-2/checkpoint_199_epoch.pkl'

    vis_num = 1000
    mask_thres = 0.5
    ##########################################################

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed()  # 设置随机种子
    in_size = 224
    # img_pil = demo_img.convert('RGB')
    original_img = np.array(img_pil)
    h, w, _ = img_pil.shape
    # img_pil = img_pil.resize((in_size, in_size), Image.BILINEAR)
    img_pil = cv2.resize(img_pil, (in_size, in_size), interpolation=cv2.INTER_AREA)

    img_hwc = np.array(img_pil)
    img_chw = img_hwc.transpose((2, 0, 1))
    img_chw = torch.from_numpy(img_chw).float()

    net = UNet(in_channels=3, out_channels=1, init_features=32)  # init_features is 64 in stander uent
    net.to(device)
    if checkpoint_load is not None:
        path_checkpoint = checkpoint_load
        checkpoint = torch.load(path_checkpoint)

        net.load_state_dict(checkpoint['model_state_dict'])
        # print('load checkpoint from %s' % path_checkpoint)
    else:
        raise Exception("\nPlease specify the checkpoint")

    net.eval()
    with torch.no_grad():

        inputs = img_chw.to(device).unsqueeze(0)
        outputs = net(inputs)

        pred = (outputs.cpu().data.numpy() * 255).astype("uint8")
        pred_gray = pred.squeeze()

        mask_pred = outputs.ge(mask_thres).cpu().data.numpy()
        mask_pred_gray = (mask_pred.squeeze() * 255).astype("uint8")

        img_hwc = inputs.cpu().data.numpy()[0, :, :, :].transpose((1, 2, 0)).astype("uint8")

    # plt.subplot(131).imshow(img_hwc)
    # plt.subplot(132).imshow(mask_pred_gray, cmap="gray")
    # plt.subplot(133).imshow(pred_gray, cmap="gray")
    # plt.show()

    # img_hwc = Image.fromarray(img_hwc)
    # img_hwc = img_hwc.resize((w, h), Image.BILINEAR)
    # img_hwc = np.array(img_hwc)
    mask_pred_gray = Image.fromarray(mask_pred_gray)
    mask_pred_gray = mask_pred_gray.resize((w, h), Image.BILINEAR)
    mask_pred_gray = np.array(mask_pred_gray)

    return original_img, mask_pred_gray

def video_replace(img, mask_bw, sky):
    sz = img.shape
    sky_resize = cv2.resize(sky, (sz[1], sz[0]))
    # mask_bw=mask_bw/255
    mask_sky = cv2.merge([mask_bw, mask_bw, mask_bw])
    mask_sky = mask_sky / 255
    mask_img = 1 - mask_sky
    I_rep = img * mask_img + sky_resize * mask_sky
    I_rep = I_rep.astype(np.uint8)
    return I_rep
