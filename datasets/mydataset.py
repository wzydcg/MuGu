#!/usr/bin/python3
# -*- coding: utf-8 -*
import cv2
from torch.utils.data import Dataset
import torch
from PIL import Image
import os
import numpy as np
from torchvision import transforms


class Mydataset(Dataset):

    CHANNELS_NUM = 1
    NUM_CLASSES = 2

    def __init__(self, mode, transform=None, target_transform=None, BASE_PATH=""):
        print(mode)
        self.items_image, self.items_mask = make_dataset(mode, BASE_PATH)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.items_image)

    def __str__(self):
        return 'Mydataset'

    def __getitem__(self, index):
        image_path = self.items_image[index]
        mask_path = self.items_mask[index]

        image = Image.open(image_path).convert('L')

        mask = cv2.imread(mask_path, 0)
        mask = mask / 255
        mask = mask.astype(np.uint8)
        mask = torch.from_numpy(mask)
        mask = mask.to(torch.uint8)
        if self.transform:
            image = self.transform(image)

        return image, mask

def make_dataset(mode, base_path):
    print(mode)
    image_path = os.path.join(base_path, "image")
    mask_path = os.path.join(base_path, "mask")
    # print(image_path)#用于调试
    image_list = []
    for file in os.listdir(image_path):
        image_list.append(os.path.join(image_path, file))

    mask_list = []
    for file in os.listdir(mask_path):
        mask_list.append(os.path.join(mask_path, file))

    # print(image_list)
    return image_list, mask_list

#用于测试
if __name__ == "__main__":
    # image_path=r"D:\Code\lungseg\processed_data\train\image\1.png"
    # image=cv2.imread(image_path)
    # cv2.imshow('tmp',image)
    # cv2.waitKey(0)
    #测试代码
    # lung_dataset = Data_Loader('/tmp/pycharm_project_186/Dataset')
    transform = transforms.Compose([transforms.ToTensor()])
    target_transform = transforms.Compose([transforms.ToTensor()])
    lung_dataset=Mydataset(mode='train',transform=transforms,target_transform=target_transform,BASE_PATH=r"D:\Code\lungseg\processed_data2\train")
    print("the number of data:",len(lung_dataset))
    # train_loader = torch.utils.data.DataLoader(dataset=lung_dataset,
    #                                            batch_size=2,
    #                                            shuffle=True)  # 每次迭代 数据洗牌
    #
    # for image, label in train_loader:  # 这里的2,1,512,512 是两个图片/一个通道/大小512*512
    #     print(image.shape)
