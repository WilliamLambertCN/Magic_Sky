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
import argparse
import cv2
import func
import unet_infer as unet

print('Video process launching...')
tic_total = time.time()

sky_num = str(5)
sky_path = 'd:/MyLearning/DIP/Final_Project/Unet/sky/' + sky_num + '.jpg'
sky = cv2.imread(sky_path)
sky = cv2.cvtColor(sky, cv2.COLOR_BGR2RGB)

video = cv2.VideoCapture("d:/MyLearning/DIP/Final_Project/Unet/Demo/video_test.mp4")
fps = video.get(cv2.CAP_PROP_FPS)
frameCount = video.get(cv2.CAP_PROP_FRAME_COUNT)
size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

videoWriter = cv2.VideoWriter('d:/MyLearning/DIP/Final_Project/Unet/Demo/trans.mp4', cv2.VideoWriter_fourcc(*'MP4V'),
                              fps, size)
success, frame = video.read()
index = 1
while index < 4:
    print('The', index, 'th frame of', int(frameCount), '...')
    # cv2.putText(frame, 'fps: ' + str(fps), (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,255), 5)
    # cv2.putText(frame, 'count: ' + str(frameCount), (0, 300), cv2.FONT_HERSHEY_SIMPLEX,2, (255,0,255), 5)
    # cv2.putText(frame, 'frame: ' + str(index), (0, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,255), 5)
    # cv2.putText(frame, 'time: ' + str(round(index / 24.0, 2)) + "s", (0,500), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,255), 5)

    # frame_pil = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))  #opencv转PIL.image
    frame = frame[..., ::-1]

    tic = time.time()

    img, mask_bw = func.video_infer(frame)

    print('infer:', time.time() - tic)

    I_rep = func.video_replace(img, mask_bw, sky)

    print('replace:', time.time() - tic)

    transfer = func.color_transfer(sky, mask_bw, I_rep, 1)

    print('color transfer:', time.time() - tic)

    mask_edge = cv2.Canny(mask_bw, 100, 200)
    mask_edge_hwc = cv2.merge([mask_edge, mask_edge, mask_edge])
    frame_cv2_out = func.guideFilter(img, transfer, mask_edge_hwc, (8, 8), 0.01)
    frame = cv2.cvtColor(np.asarray(frame_cv2_out), cv2.COLOR_RGB2BGR)  # PIL.image转opencv
    # cv2.imwrite('d:/MyLearning/DIP/Final_Project/Unet/Demo/' + 'frame_' + str(index) + '.png', frame) #存储为图像,保存名为 文件夹名_数字（第几个文件）.jpg
    # frame = cv2.cvtColor(np.asarray(frame_pil),cv2.COLOR_RGB2BGR) #PIL.image转opencv
    print('filter:', time.time() - tic)

    # cv2.imshow("new video", frame)
    # cv2.waitKey(int(1000 / int(fps)))
    videoWriter.write(frame)
    success, frame = video.read()
    index += 1

video.release()
toc = time.time()
print('Done! Total time is ' + str(toc - tic_total) + 's')
