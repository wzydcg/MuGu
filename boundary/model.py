#!/usr/bin/python3
# -*- coding: utf-8 -*
import torch
from torch import nn
from models.network_blocks import DoubleConvBlock
from models.aspp import ASPP
from boundary.BANet import PEE
from models.esa_modules import ESA_block

class BPB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels * 5, in_channels, kernel_size=1, dilation=1, bias=False).to('cpu')
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, dilation=1, bias=False).to('cpu')
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=1, padding=1, bias=False).to('cpu')
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=2, padding=2, bias=False).to('cpu')
        self.conv4 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=4, padding=4, bias=False).to('cpu')
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=6, padding=6, bias=False).to('cpu')
        self.s = nn.Sigmoid().to('cpu')
    def forward(self, image):
        x = torch.cat([self.conv1(image),self.conv2(image),self.conv3(image),self.conv4(image),self.conv5(image)],dim=1)
        return self.s(self.conv(x))

class UNet_border(nn.Module):

    def __init__(self, num_classes, in_channels):
        super().__init__()
        nb_filters = [32, 64, 128, 256, 512]
        # self.esa = ESA_block(nb_filters[4])
        self.aspp = ASPP(nb_filters[4], nb_filters[4], [6,12,18])
        # self.acmix = ACmix(in_planes = nb_filters[4], out_planes = nb_filters[4])
        # 标准的2倍上采样和下采样，因为没有可以学习的参数，可以共享
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # 下采样的模块
        self.conv0_0 = DoubleConvBlock(in_channels, nb_filters[0], nb_filters[0])
        self.conv1_0 = DoubleConvBlock(nb_filters[0], nb_filters[1], nb_filters[1])
        self.conv2_0 = DoubleConvBlock(nb_filters[1], nb_filters[2], nb_filters[2])
        self.conv3_0 = DoubleConvBlock(nb_filters[2], nb_filters[3], nb_filters[3])
        self.conv4_0 = DoubleConvBlock(nb_filters[3], nb_filters[4], nb_filters[4])
        # PEE 模块
        self.pee0 = PEE(nb_filters[0],nb_filters[0],[3,5,7])
        self.pee1 = PEE(nb_filters[1],nb_filters[1],[3,5,7])
        self.pee2 = PEE(nb_filters[2],nb_filters[2],[3,5,7])
        self.pee3 = PEE(nb_filters[3],nb_filters[3],[3,5,7])

        # 上采样的模块
        self.conv3_1 = DoubleConvBlock(nb_filters[4] + nb_filters[3], nb_filters[3], nb_filters[3])
        self.conv2_2 = DoubleConvBlock(nb_filters[3] + nb_filters[2], nb_filters[2], nb_filters[2])
        self.conv1_3 = DoubleConvBlock(nb_filters[2] + nb_filters[1], nb_filters[1], nb_filters[1])
        self.conv0_4 = DoubleConvBlock(nb_filters[1] + nb_filters[0], nb_filters[0], nb_filters[0])

        self.conv3_2 = DoubleConvBlock(nb_filters[3], nb_filters[2], nb_filters[2])
        self.conv2_1 = DoubleConvBlock(nb_filters[2], nb_filters[1], nb_filters[1])
        self.conv1_1 = DoubleConvBlock(nb_filters[1], nb_filters[0], nb_filters[0])

        # BPB 模块
        self.b1 = BPB(32)
        self.b2 = BPB(64)
        self.b3 = BPB(128)
        self.b4 = BPB(256)
        self.b5 = BPB(512)

        self.s = nn.Sigmoid()
        # 最后接一个Conv计算在所有类别上的分数
        self.final = nn.Conv2d(nb_filters[0], num_classes, kernel_size=1, stride=1)

    def forward(self, input):
        # 下采样编码
        x0_0 = self.conv0_0(input)
        F0m= self.pee0(x0_0)
        M1 = self.b1(x0_0)
        x0_0 = x0_0 + x0_0 * M1

        x1_0 = self.conv1_0(self.down(x0_0))
        F1m = self.pee1(x1_0)

        M2 = self.b2(x1_0)
        x1_0 = x1_0 + x1_0 * M2

        x2_0 = self.conv2_0(self.down(x1_0))
        F2m = self.pee2(x2_0)

        M3 = self.b3(x2_0)
        x2_0 = x2_0 + x2_0 * M3

        x3_0 = self.conv3_0(self.down(x2_0))
        F3m= self.pee3(x3_0)

        M4 = self.b4(x3_0)
        x3_0 = x3_0 + x3_0 * M4

        x4_0 = self.conv4_0(self.down(x3_0))
        M5 = self.b5(x4_0)
        x4_0 = self.aspp(x4_0 + x4_0 * M5)

        # 特征融合并进行上采样解码，使用concatenate进行特征融合
        x3_0 = F3m + (1 - self.s(F3m)) * (self.conv3_0(self.down(self.s(F2m) * F2m)))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))
        M6 = self.b4(x3_1)
        x3_1 = x3_1 + x3_1 * M6

        x2_0 = F2m + (1 - self.s(F2m)) * (self.conv2_0(self.down(self.s(F1m) * F1m)) + self.conv3_2(self.up(self.s(F3m) * F3m)))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], dim=1))
        M7 = self.b3(x2_2)
        x2_2 = x2_2 + x2_2 * M7

        x1_0 = F1m + (1 - self.s(F1m)) * (self.conv1_0(self.down(self.s(F0m) * F0m)) + self.conv2_1(self.up(self.s(F2m) * F2m)))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], dim=1))
        M8 = self.b2(x1_3)
        x1_3 = x1_3 + x1_3 * M8

        x0_0 = F0m + (1 - self.s(F0m)) * (self.conv1_1(self.up(self.s(F1m) * F1m)))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], dim=1))
        M9 = self.b1(x0_4)
        x0_4 = x0_4 + x0_4 * M9
        # 计算每个类别上的得分
        output = self.final(x0_4)

        return output,M1,M2,M3,M4,M5,M6,M7,M8,M9


if __name__ == '__main__':
    x=torch.rand([4, 3, 512, 512])
    # avg = nn.AvgPool2d(3)
    # print(avg(x).shape)

    net = UNet_border(2,3)
    res,_,_,_,_,_,_,_,_,_ = net(x)
    print(res.shape)

    # net=UNet(num_classes=2, in_channels=3)
    # print(net(x).shape)