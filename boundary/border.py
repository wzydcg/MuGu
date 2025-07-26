import numpy as np
import cv2 as cv
from skimage import io, color, util, morphology, filters, measure
from scipy import ndimage
import matplotlib.pyplot as plt

def border(img):
    # 使用椭圆核进行膨胀
    img = img.astype(np.uint8)  # 确保图像为无符号8位整型
    # kernel_ellipse = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    # img_dilate = cv.dilate(img, kernel_ellipse)
    kernel_square = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    border = img - cv.erode(img, kernel_square, iterations=1)
    return border
# # 用于测试
# img = cv.imread('D:/Code/lungseg/processed_data/train/mask/1.png', cv.IMREAD_GRAYSCALE)
# img1 = cv.imread('D:/Code/lungseg/processed_data/train/image/1.png',cv.IMREAD_GRAYSCALE)
#
# # 使用椭圆核进行膨胀
# kernel_ellipse = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
# img_dilate = cv.dilate(img, kernel_ellipse)
#
# kernel_square = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
# img_erode1 = img_dilate - cv.morphologyEx(img, cv.MORPH_ERODE, kernel_square)
#
# kernel_ellipse = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
# img_erode2 = img_dilate - cv.erode(img, kernel_ellipse)
#
# # 显示结果
# plt.figure(figsize=(15, 7))
# plt.gray()
#
# # 原图像
# plt.subplot(1, 3, 1);
# plt.imshow(img1)
# plt.title('Original image')
# plt.axis('off')
# # 3*3正方形结构元素腐蚀结果
# plt.subplot(1, 3, 2);
# plt.imshow(img_erode1 * img1)
# plt.title('Eroded by 3*3 square')
# plt.axis('off')
# # 15*15的椭圆形结构元素腐蚀结果
# plt.subplot(1, 3, 3);
# plt.imshow(img_erode2 * img1)
# plt.title('Eroded by 5*5 ellipse')
# plt.axis('off')
# plt.show()
