#!/usr/bin/python3
# -*- coding: utf-8 -*
import torch
from torch import nn
from models.network_blocks import DoubleConvBlock
from models.esa_modules import ESA_block


class UNet(nn.Module):

    def __init__(self, num_classes, in_channels, is_esa=False):
        super().__init__()
        nb_filters = [32, 64, 128, 256, 512]
        self.is_esa = is_esa
        self.is_ese_1_0 = ESA_block(nb_filters[1])
        self.is_ese_2_0 = ESA_block(nb_filters[2])
        self.is_ese_3_0 = ESA_block(nb_filters[3])
        self.is_ese_4_0 = ESA_block(nb_filters[4])

        # 标准的2倍上采样和下采样，因为没有可以学习的参数，可以共享
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # 下采样的模块
        self.conv0_0 = DoubleConvBlock(in_channels, nb_filters[0], nb_filters[0])
        self.conv1_0 = DoubleConvBlock(nb_filters[0], nb_filters[1], nb_filters[1])
        self.conv2_0 = DoubleConvBlock(nb_filters[1], nb_filters[2], nb_filters[2])
        self.conv3_0 = DoubleConvBlock(nb_filters[2], nb_filters[3], nb_filters[3])
        self.conv4_0 = DoubleConvBlock(nb_filters[3], nb_filters[4], nb_filters[4])

        # 上采样的模块
        self.conv3_1 = DoubleConvBlock(nb_filters[4] + nb_filters[3], nb_filters[3], nb_filters[3])
        self.conv2_2 = DoubleConvBlock(nb_filters[3] + nb_filters[2], nb_filters[2], nb_filters[2])
        self.conv1_3 = DoubleConvBlock(nb_filters[2] + nb_filters[1], nb_filters[1], nb_filters[1])
        self.conv0_4 = DoubleConvBlock(nb_filters[1] + nb_filters[0], nb_filters[0], nb_filters[0])

        # 最后接一个Conv计算在所有类别上的分数
        self.final = nn.Conv2d(nb_filters[0], num_classes, kernel_size=1, stride=1)

    def forward(self, input):
        # 下采样编码
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.down(x0_0))
        if self.is_esa:
            x1_0 = x1_0 + self.is_ese_1_0(x1_0)

        x2_0 = self.conv2_0(self.down(x1_0))
        if self.is_esa:
            x2_0 = x2_0 + self.is_ese_2_0(x2_0)

        x3_0 = self.conv3_0(self.down(x2_0))
        if self.is_esa:
            x3_0 = x3_0 + self.is_ese_3_0(x3_0)

        x4_0 = self.conv4_0(self.down(x3_0))
        if self.is_esa:
            x4_0 = x4_0 + self.is_ese_4_0(x4_0)

        # 特征融合并进行上采样解码，使用concatenate进行特征融合
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], dim=1))

        # 计算每个类别上的得分
        output = self.final(x0_4)

        return output
if __name__ == '__main__':
    x=torch.rand([4, 3, 512, 512])
    net=UNet(num_classes=2, in_channels=3)
    print(net(x).shape)