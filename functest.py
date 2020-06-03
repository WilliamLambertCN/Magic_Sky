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
from color_transfer import color_transfer
import argparse
import cv2
import func
import unet_infer as unet

# img_num=str(5)
# img_path='d:/MyLearning/DIP/Final_Project/Unet/Demo/'+img_num+'.png'
# # demo_img = Image.open(img_path).convert('RGB')
# demo_img = Image.open(img_path)
# img_hwc,mask_pred_gray=func.infer(demo_img)
