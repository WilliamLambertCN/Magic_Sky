# author: 
# contact: https://github.com/WilliamLambertCN
# datetime:2020/5/28 14:07
"""
文件说明： Unet inference
"""
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
def unet_infer(demo_path_img,demo,save_result):
    # demo = True
    # demo_path_img = 'd:/MyLearning/DIP/Final_Project/Unet/Demo/1.jpg'
    # save_result = True

    testset_path = os.path.join("dataset/testset")
    checkpoint_load = 'd:/MyLearning/DIP/Final_Project/Unet/test2_lovasz_1e-2/checkpoint_199_epoch.pkl'
    shuffle_dataset = True

    vis_num = 1000
    mask_thres = 0.5
    ##########################################################

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed()  # 设置随机种子
    in_size = 224

    if not demo:
        testset = SkyDataset(testset_path)
        valid_loader = DataLoader(testset, batch_size=1, drop_last=False, shuffle=False)
    else:
        img_pil = Image.open(demo_path_img).convert('RGB')
        original_img=np.array(img_pil)
        w, h = img_pil.size
        img_pil = img_pil.resize((in_size, in_size), Image.BILINEAR)

        img_hwc = np.array(img_pil)
        img_chw = img_hwc.transpose((2, 0, 1))
        img_chw = torch.from_numpy(img_chw).float()

    net = UNet(in_channels=3, out_channels=1, init_features=32)  # init_features is 64 in stander uent
    net.to(device)
    if checkpoint_load is not None:
        path_checkpoint = checkpoint_load
        checkpoint = torch.load(path_checkpoint)

        net.load_state_dict(checkpoint['model_state_dict'])
        print('load checkpoint from %s' % path_checkpoint)
    else:
        raise Exception("\nPlease specify the checkpoint")

    net.eval()
    with torch.no_grad():
        if not demo:
            for idx, (inputs, labels) in enumerate(valid_loader):
                if idx > vis_num:
                    break
                if torch.cuda.is_available():
                    inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)

                pred = (outputs.cpu().data.numpy() * 255).astype("uint8")
                pred_gray = pred.squeeze()

                mask_pred = outputs.ge(mask_thres).cpu().data.numpy()
                mask_pred_gray = (mask_pred.squeeze() * 255).astype("uint8")

                print('idx>>%d, Dice>>%.4f' % (idx, compute_dice(mask_pred, labels.cpu().numpy())))
                img_hwc = inputs.cpu().data.numpy()[0, :, :, :].transpose((1, 2, 0)).astype("uint8")
                img_label = (labels.cpu().data.numpy()[0, 0, :, :] * 255).astype("uint8")
                plt.subplot(221).imshow(img_hwc)
                plt.title('%d Original IMG' % idx)
                plt.subplot(222).imshow(img_label, cmap="gray")
                plt.title('%d Original Label' % idx)
                plt.subplot(223).imshow(mask_pred_gray, cmap="gray")
                plt.title('%d Binary Label' % idx)
                plt.subplot(224).imshow(pred_gray, cmap="gray")
                plt.title('%d Raw Label' % idx)
                plt.tight_layout()
                plt.savefig('results/%d_img' % idx)
                plt.show()
                plt.close()
                if save_result:
                    pred_gray_img = Image.fromarray(pred_gray)
                    pred_gray_img.save('results/%d_pred_gray_img.png' % idx)

                    img_hwc_img = Image.fromarray(img_hwc)
                    img_hwc_img.save('results/%d_img_hwc.png' % idx)
        else:
            inputs = img_chw.to(device).unsqueeze(0)
            outputs = net(inputs)

            pred = (outputs.cpu().data.numpy() * 255).astype("uint8")
            pred_gray = pred.squeeze()

            mask_pred = outputs.ge(mask_thres).cpu().data.numpy()
            mask_pred_gray = (mask_pred.squeeze() * 255).astype("uint8")

            img_hwc = inputs.cpu().data.numpy()[0, :, :, :].transpose((1, 2, 0)).astype("uint8")

            if save_result:
                pred_gray_img = Image.fromarray(pred_gray)
                pred_gray_img = pred_gray_img.resize((w, h), Image.BICUBIC)
                pred_gray_img.save('d:/MyLearning/DIP/Final_Project/Unet/results/1_pred_gray_img.png')
                mask_pred_gray_img = Image.fromarray(mask_pred_gray)
                mask_pred_gray_img = mask_pred_gray_img.resize((w, h), Image.BICUBIC)
                mask_pred_gray_img.save('d:/MyLearning/DIP/Final_Project/Unet/results/1_mask_pred_gray_img.png')
                img_hwc_img = Image.open(demo_path_img).convert('RGB')
                img_hwc_img.save('d:/MyLearning/DIP/Final_Project/Unet/results/1_img_hwc_img.png')
            # plt.subplot(131).imshow(img_hwc)
            # plt.subplot(132).imshow(mask_pred_gray, cmap="gray")
            # plt.subplot(133).imshow(pred_gray, cmap="gray")
            # plt.show()
            # plt.pause(0.5)
            # plt.close()
           
            # img_hwc = Image.fromarray(img_hwc)
            # img_hwc = img_hwc.resize((w, h), Image.BILINEAR)
            # img_hwc = np.array(img_hwc)
            mask_pred_gray = Image.fromarray(mask_pred_gray)
            mask_pred_gray = mask_pred_gray.resize((w, h), Image.BILINEAR)
            mask_pred_gray = np.array(mask_pred_gray)

            return original_img,mask_pred_gray
