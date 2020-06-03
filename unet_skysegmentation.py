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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed()  # 设置随机种子

def compute_dice(y_pred, y_true):
    """
    :param y_pred: 4-d tensor, value = [0,1]
    :param y_true: 4-d tensor, value = [0,1]
    :return: Dice index 2*TP/(2*TP+FP+FN)=2TP/(pred_P+true_P)
    """
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    y_pred, y_true = np.round(y_pred).astype(int), np.round(y_true).astype(int)
    return np.sum(y_pred[y_true == 1]) * 2.0 / (np.sum(y_pred) + np.sum(y_true))

if __name__ == "__main__":

    # config
    LR = 0.01
    BATCH_SIZE = 20
    max_epoch = 200  # 400
    start_epoch = 0
    lr_step = 100
    val_interval = 1
    checkpoint_interval = 20
    vis_num = 10
    mask_thres = 0.5
    # Tensorboard计数
    iter_count = 0
    logdir = './test4_lovasz_1e-2'
    writer = SummaryWriter(log_dir=logdir)
    ##########预训练与否#############
    pretrained = True
    checkpoint_load = 'test3_lovasz_1e-2/checkpoint_99_epoch.pkl'

    trainset_path = os.path.join("dataset/trainset")
    testset_path = os.path.join("dataset/testset")
    #########是否预加载############
    if pretrained == False:
        checkpoint_load = None
    else:
        print('Loaded checkpoint from %s.' % checkpoint_load)

    # step 1 划分训练集、验证集
    trainset = SkyDataset(trainset_path)
    testset = SkyDataset(testset_path)

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE,
                              drop_last=False, shuffle=True)
    valid_loader = DataLoader(testset, batch_size=1,
                              drop_last=False, shuffle=False)

    # step 2
    net = UNet(in_channels=3, out_channels=1, init_features=32)  # init_features is 64 in stander uent

    net.to(device)

    # step 3
    # loss_fn = nn.MSELoss()
    loss_fn = lovasz_hinge

    # step 4
    optimizer = optim.SGD(net.parameters(), momentum=0.9, lr=LR, weight_decay=1e-2)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=0.2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10,
                                                           verbose=True, threshold=1e-2, threshold_mode='rel',
                                                           cooldown=0, min_lr=1e-5, eps=1e-8)
    ################################################################################################################
    if checkpoint_load is not None:
        path_checkpoint = checkpoint_load
        checkpoint = torch.load(path_checkpoint)

        net.load_state_dict(checkpoint['model_state_dict'])

        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # start_epoch = checkpoint['epoch']
        # scheduler.last_epoch = start_epoch
    ##################################################################################################################
    # step 5
    train_curve = list()
    valid_curve = list()
    train_dice_curve = list()
    valid_dice_curve = list()
    for epoch in range(start_epoch, max_epoch):

        train_loss_total = 0.
        train_dice_total = 0.
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        net.train()
        for iter, (inputs, labels) in enumerate(train_loader):

            iter_count += 1
            if torch.cuda.is_available():
                inputs, labels = inputs.to(device), labels.to(device)

            # forward
            outputs = net(inputs)

            # backward
            optimizer.zero_grad()
            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()

            # print
            train_dice = compute_dice(outputs.ge(mask_thres).cpu().data.numpy(), labels.cpu())
            train_dice_curve.append(train_dice)
            train_curve.append(loss.item())

            train_loss_total += loss.item()

            writer.add_scalar("Train Loss", train_loss_total / (iter + 1), iter_count)
            writer.add_scalar("Train Dice", train_dice, iter_count)
            print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] running_loss: {:.4f}, mean_loss: {:.4f} "
                  "running_dice: {:.4f} lr:{}".format(epoch, max_epoch, iter + 1, len(train_loader), loss.item(),
                                                      train_loss_total / (iter + 1), train_dice,
                                                      optimizer.state_dict()['param_groups'][0]['lr']))

        scheduler.step(train_loss_total / (iter + 1))

        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint = {"model_state_dict": net.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch}
            path_checkpoint = os.path.join(logdir, "checkpoint_{}_epoch.pkl".format(epoch))
            torch.save(checkpoint, path_checkpoint)

        # validate the model
        if (epoch + 1) % val_interval == 0:

            net.eval()
            valid_loss_total = 0.
            valid_dice_total = 0.

            with torch.no_grad():
                for j, (inputs, labels) in enumerate(valid_loader):
                    if torch.cuda.is_available():
                        inputs, labels = inputs.to(device), labels.to(device)

                    outputs = net(inputs)
                    loss = loss_fn(outputs, labels)

                    valid_loss_total += loss.item()

                    valid_dice = compute_dice(outputs.ge(mask_thres).cpu().data, labels.cpu())
                    valid_dice_total += valid_dice

                valid_loss_mean = valid_loss_total / len(valid_loader)
                valid_dice_mean = valid_dice_total / len(valid_loader)
                valid_curve.append(valid_loss_mean)
                valid_dice_curve.append(valid_dice_mean)

                writer.add_scalar("Valid Loss", valid_loss_mean, iter_count)
                writer.add_scalar("Valid Dice", valid_dice_mean, iter_count)
                print("Valid:\t Epoch[{:0>3}/{:0>3}] mean_loss: {:.4f} dice_mean: {:.4f}".format(
                        epoch, max_epoch, valid_loss_mean, valid_dice_mean))

        for name, param in net.named_parameters():
            writer.add_histogram(name + '_grad', param.grad, epoch)
            writer.add_histogram(name + '_data', param, epoch)

    ###################################################################################################################
    # 可视化
    # valid_dir = os.path.join(BASE_DIR, "../../..", "data", "PortraitDataset", "valid")
    # valid_set = SkyDataset(valid_dir)
    # valid_loader = DataLoader(valid_set, batch_size=1, shuffle=True, drop_last=False)
    net.eval()
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(valid_loader):
            if idx > vis_num:
                break
            if torch.cuda.is_available():
                inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            pred = outputs.ge(mask_thres)

            mask_pred = outputs.ge(0.5).cpu().data.numpy().astype("uint8")

            img_hwc = inputs.cpu().data.numpy()[0, :, :, :].transpose((1, 2, 0)).astype("uint8")
            plt.subplot(121).imshow(img_hwc)
            mask_pred_gray = mask_pred.squeeze() * 255
            plt.subplot(122).imshow(mask_pred_gray, cmap="gray")
            plt.show()
            plt.pause(0.5)
            plt.close()

    # plot curve
    train_x = range(len(train_curve))
    train_y = train_curve

    train_iters = len(train_loader)
    valid_x = np.arange(1, len(
            valid_curve) + 1) * train_iters * val_interval  # 由于valid中记录的是epochloss，需要对记录点进行转换到iterations
    valid_y = valid_curve

    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.legend(loc='upper right')
    plt.ylabel('loss value')
    plt.xlabel('Iteration')
    plt.title("Plot in {} epochs".format(max_epoch))
    plt.show()

    # dice curve
    train_x = range(len(train_dice_curve))
    train_y = train_dice_curve

    train_iters = len(train_loader)
    valid_x = np.arange(1, len(
            valid_dice_curve) + 1) * train_iters * val_interval  # 由于valid中记录的是epochloss，需要对记录点进行转换到iterations
    valid_y = valid_dice_curve

    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.legend(loc='upper right')
    plt.ylabel('dice value')
    plt.xlabel('Iteration')
    plt.title("Plot in {} epochs".format(max_epoch))
    plt.show()
    torch.cuda.empty_cache()
