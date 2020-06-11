import cv2
import matplotlib.pyplot as plt
import glob
import time
import numpy as np

img = cv2.imread('ly.png')

rows, cols, channels = img.shape
plt.imshow(img[..., ::-1])
plt.show()

# 转换hsv
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_blue = np.array([90, 60, 60])
upper_blue = np.array([110, 255, 255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)
# cv2.imshow('Mask', mask)

# 腐蚀膨胀
kernel = np.ones((11, 11), np.uint8)
erode = cv2.erode(mask, kernel, iterations=1)
plt.imshow(erode[..., ::-1])
plt.show()
# cv2.imshow('erode', erode)
dilate = cv2.dilate(erode, kernel, iterations=1)
plt.imshow(dilate[..., ::-1])
plt.show()
# cv2.imshow('dilate', dilate)

# 遍历替换
for i in range(rows):
    for j in range(cols):
        if dilate[i, j] == 255:
            img[i, j] = (255, 255, 255)  # 此处替换颜色，为BGR通道
plt.imshow(img[..., ::-1])
plt.show()
