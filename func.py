import os
import torch
import numpy as np
from PIL import Image
from tools.common_tools import set_seed
from tools.unet import UNet
import cv2

def find_sky_rect(mask):
    """

    Args:
        mask:

    Returns:

    """
    index = np.where(mask != 0)
    index = np.array(index, dtype=int)
    x = index[0, :]
    y = index[1, :]
    r2 = np.min(x)
    r1 = np.max(x)
    c2 = np.min(y)
    c1 = np.max(y)
    return (r1, c1, r2, c2)

def guideFilter(I, p, mask_edge, winSize, eps):  # input p,giude I
    """

    Args:
        I:
        p:
        mask_edge:
        winSize:
        eps:

    Returns:

    """

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
    kernel = np.ones((5, 5), np.uint8)
    # kernel=np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]],np.uint8)
    edge = cv2.dilate(mask_edge, kernel)

    # edge8=edge*255
    # edge8=edge8.astype(np.uint8)
    # mask_edge8=mask_edge*255
    # mask_edge8=mask_edge8.astype(np.uint8)
    # cv2.imwrite('d:/MyLearning/DIP/Final_Project/Report/mask_edge.png',mask_edge8)
    # cv2.imwrite('d:/MyLearning/DIP/Final_Project/Report/edge.png',edge8)

    q[edge == 1] = mean_a[edge == 1] * I[edge == 1] + mean_b[edge == 1]

    q = q * 255
    q[q > 255] = 255
    q = np.round(q)
    q = q.astype(np.uint8)
    return q

def color_transfer(source, mask_bw, target, mode):
    """

    Args:
        source:
        mask_bw:
        target:
        mode:

    Returns:

    """
    source = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype("float32")

    (l, a, b) = cv2.split(target)
    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
    if mode:
        index = np.where(mask_bw == 0)
        index = np.array(index, dtype=int)
        sz = index.shape

        fl = target[:, :, 0][mask_bw == 0]
        fa = target[:, :, 1][mask_bw == 0]
        fb = target[:, :, 2][mask_bw == 0]

        (lMeanTar_1, lStdTar_1) = (np.mean(fl), np.std(fl, ddof=1))
        (aMeanTar_1, aStdTar_1) = (np.mean(fa), np.std(fa, ddof=1))
        (bMeanTar_1, bStdTar_1) = (np.mean(fb), np.std(fb, ddof=1))
        (lMeanTar_2, lStdTar_2, aMeanTar_2, aStdTar_2, bMeanTar_2, bStdTar_2) = image_stats(source)
        # (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)

        l[mask_bw == 0] -= lMeanTar_1
        a[mask_bw == 0] -= aMeanTar_1
        b[mask_bw == 0] -= bMeanTar_1

        alpha = 0.5
        beta = 1 - alpha

        l[mask_bw == 0] *= (alpha * lStdSrc + beta * lStdTar_1) / lStdTar_1
        a[mask_bw == 0] *= (alpha * aStdSrc + beta * aStdTar_1) / aStdTar_1
        b[mask_bw == 0] *= (alpha * bStdSrc + beta * bStdTar_1) / bStdTar_1

        l[mask_bw == 0] += alpha * lMeanSrc + beta * lMeanTar_1
        a[mask_bw == 0] += alpha * aMeanSrc + beta * aMeanTar_1
        b[mask_bw == 0] += alpha * bMeanSrc + beta * bMeanTar_1

    else:
        (lMeanTar_2, lStdTar_2, aMeanTar_2, aStdTar_2, bMeanTar_2, bStdTar_2) = image_stats(target)
        l -= lMeanTar_2
        a -= aMeanTar_2
        b -= bMeanTar_2

        l = (lStdSrc / lStdTar_2) * l
        a = (aStdSrc / aStdTar_2) * a
        b = (bStdSrc / bStdTar_2) * b
        l += lMeanSrc
        a += aMeanSrc
        b += bMeanSrc

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

def photo_infer(src, net):
    """
    Args:
        src:
        net：
    Returns: pred mask
    """
    # checkpoint_load = 'tools/checkpoint_199_epoch.pkl'
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # set_seed()  # 设置随机种子
    mask_thres = 0.5
    in_size = 224
    h, w, c = src.shape
    # print(h, w)
    img_hwc = cv2.resize(src, (in_size, in_size), interpolation=cv2.INTER_AREA)
    img_chw = img_hwc.transpose((2, 0, 1))
    img_chw = torch.from_numpy(img_chw).float()
    # net = UNet(in_channels=3, out_channels=1, init_features=32)  # init_features is 64 in stander uent
    # net.to(device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        inputs = img_chw.to(device).unsqueeze(0)
        outputs = net(inputs)
        mask = (outputs.ge(mask_thres).cpu().data.numpy() * 255).squeeze().astype("uint8")
    mask = cv2.resize(mask, (w, h))
    # print("Input's shape is ", mask.shape)
    return mask

def photo_replace(src, tgt, net, mode=0):
    """

    Args:
        srcname:
        tgtname:
    Returns: results

    """

    mask = photo_infer(src, net)
    if mode == 0:
        I_rep = replace_sky(src, mask, tgt)
    else:
        I_rep = video_replace(src, mask, tgt)
    # print('color transferring...')
    transfer = color_transfer(tgt, mask, I_rep, 1)
    mask_edge = cv2.Canny(mask, 100, 200)
    mask_edge_hwc = cv2.merge([mask_edge, mask_edge, mask_edge])
    # print('guide filtering...')
    result = guideFilter(src, transfer, mask_edge_hwc, (8, 8), 0.01)
    return result

def photo_improve(src, mask, tgt):
    """

    Args:
        srcname:
        tgtname:
    Returns: results

    """
    I_rep = photo_replace(src, mask, tgt)
    # print('color transferring...')
    transfer = color_transfer(tgt, mask, I_rep, 1)
    mask_edge = cv2.Canny(mask, 100, 200)
    mask_edge_hwc = cv2.merge([mask_edge, mask_edge, mask_edge])
    # print('guide filtering...')
    result = guideFilter(src, transfer, mask_edge_hwc, (8, 8), 0.01)
    return result

##########################################################
def video_infer(img_pil):
    """

    Args:
        img_pil:

    Returns:

    """
    checkpoint_load = 'tools/checkpoint_199_epoch.pkl'

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

    mask_pred_gray = Image.fromarray(mask_pred_gray)
    mask_pred_gray = mask_pred_gray.resize((w, h), Image.BILINEAR)
    mask_pred_gray = np.array(mask_pred_gray)

    return original_img, mask_pred_gray

def video_replace(img, mask, sky):
    """

    Args:
        img:
        mask:
        sky:

    Returns:

    """
    (h, w, c) = img.shape
    sky_resize = cv2.resize(sky, (w, h), cv2.INTER_AREA)
    # mask=mask/255
    mask_sky = cv2.merge([mask, mask, mask])
    mask_sky = mask_sky / 255
    mask_img = 1 - mask_sky
    I_rep = img * mask_img + sky_resize * mask_sky
    I_rep = I_rep.astype(np.uint8)
    return I_rep

def replace_sky(img, mask, sky):
    """

    Args:
        img:
        mask:
        sky:

    Returns:

    """
    r1, c1, r2, c2 = find_sky_rect(mask)

    height = r1 - r2 + 1
    width = c1 - c2 + 1

    sky_resize = cv2.resize(sky, (width, height), cv2.INTER_AREA)

    I_rep = img.copy()
    index = mask[r2:r1 + 1, c2:c1 + 1] != 0
    I_rep[r2:r1 + 1, c2:c1 + 1][index] = sky_resize[index]

    return I_rep
