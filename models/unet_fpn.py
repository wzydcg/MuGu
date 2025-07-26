#!/usr/bin/python3
# -*- coding: utf-8 -*
import torch
from torch import nn
from models.network_blocks import DoubleConvBlock
from models.esa_modules import ESA_block
from models.aspp import ASPP

class UNet_improve(nn.Module):#自己根据特征金字塔的思想对UNET模型进行的修改，但是特征金字塔的结构没有出来

    def __init__(self, num_classes, in_channels, is_esa=False ,is_aspp=False):
        super().__init__()
        nb_filters = [32, 64, 128, 256, 512]#列出卷积核的尺寸
        self.is_aspp = is_aspp
        self.is_esa = is_esa

        self.is_ese_4_0 = ESA_block(nb_filters[4])#512

        # 标准的2倍上采样和下采样，因为没有可以学习的参数，可以共享
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # 下采样的模块
        self.conv0_0 = DoubleConvBlock(in_channels, nb_filters[0], nb_filters[0])#3 -> 32
        self.conv1_0 = DoubleConvBlock(nb_filters[0], nb_filters[1], nb_filters[1])#32 -> 64
        self.conv2_0 = DoubleConvBlock(nb_filters[1], nb_filters[2], nb_filters[2])#64 -> 128
        self.conv3_0 = DoubleConvBlock(nb_filters[2], nb_filters[3], nb_filters[3])#128 -> 256
        self.conv4_0 = DoubleConvBlock(nb_filters[3], nb_filters[4], nb_filters[4])#256 -> 512

        # 处理UNET和fpn模块图像之间的关联 1×1卷积
        self.conv0_1 = nn.Conv2d(nb_filters[0], nb_filters[0], kernel_size=1, stride=1)
        self.conv1_1 = nn.Conv2d(nb_filters[1], nb_filters[0], kernel_size=1, stride=1)
        self.conv2_1 = nn.Conv2d(nb_filters[2], nb_filters[0], kernel_size=1, stride=1)
        self.conv3_1 = nn.Conv2d(nb_filters[3], nb_filters[0], kernel_size=1, stride=1)
        self.conv4_1 = nn.Conv2d(nb_filters[4], nb_filters[0], kernel_size=1, stride=1)
        #3×3卷积
        self.conv2 = nn.Conv2d(nb_filters[0], nb_filters[0], kernel_size=3, padding=1, stride=1)

        # bottleneck中加入ASPP模块
        self.aspp = ASPP(nb_filters[4], nb_filters[4], [6, 12, 18])


        # 上采样的模块
        self.conv3_3 = DoubleConvBlock(nb_filters[4] + nb_filters[0], nb_filters[3], nb_filters[3])# 512+32 -> 256
        self.conv2_3 = DoubleConvBlock(nb_filters[3] + nb_filters[0], nb_filters[2], nb_filters[2])# 256+32 -> 128
        self.conv1_3 = DoubleConvBlock(nb_filters[2] + nb_filters[0], nb_filters[1], nb_filters[1])# 128+32 -> 64
        self.conv0_3 = DoubleConvBlock(nb_filters[1] + nb_filters[0], nb_filters[0], nb_filters[0])# 64+32 -> 32

        # 最后接一个Conv计算在所有类别上的分数
        self.final = nn.Conv2d(nb_filters[0], num_classes, kernel_size=1, stride=1)#32 -> 2

    def forward(self, input):
        # 下采样编码
        x0_0 = self.conv0_0(input)
        f0_0 = self.conv0_1(x0_0)

        x1_0 = self.conv1_0(self.down(x0_0))
        f1_0 = self.conv1_1(x1_0)

        x2_0 = self.conv2_0(self.down(x1_0))
        f2_0 = self.conv2_1(x2_0)

        x3_0 = self.conv3_0(self.down(x2_0))
        f3_0 = self.conv3_1(x3_0)

        x4_0 = self.conv4_0(self.down(x3_0))

        if self.is_esa:
           x4_0 = x4_0 + self.is_ese_4_0(x4_0)

        if self.is_aspp:
            x4_0 = self.aspp(x4_0)

        f4_0 = self.conv4_1(x4_0)

        f5_0 = f3_0 + self.up(f4_0)
        f6_0 = f2_0 + self.up(f5_0)
        f7_0 = f1_0 + self.up(f6_0)
        f8_0 = f0_0 + self.up(f7_0)

        # 特征融合并进行上采样解码，使用concatenate进行特征融合
        x5_0 = self.conv3_3(torch.cat([self.conv2(f5_0) ,self.up(x4_0)], dim=1))
        x6_0 = self.conv2_3(torch.cat([self.conv2(f6_0) ,self.up(x5_0)], dim=1))
        x7_0 = self.conv1_3(torch.cat([self.conv2(f7_0) ,self.up(x6_0)], dim=1))
        x8_0 = self.conv0_3(torch.cat([self.conv2(f8_0) ,self.up(x7_0)], dim=1))

        # 计算每个类别上的得分
        output = self.final(x8_0)

        return output

class UNet_fpn(nn.Module):
    def __init__(self, num_classes, in_channels, is_esa=False, is_aspp=False):
        super().__init__()
        nb_filters = [32, 64, 128, 256, 512]  # 列出卷积核的尺寸
        self.is_aspp = is_aspp
        self.is_esa = is_esa

        self.is_ese_4_0 = ESA_block(nb_filters[4])  # 512
        self.is_ese_0_0 = ESA_block(nb_filters[0])  # 32
        # 标准的2倍上采样和下采样，因为没有可以学习的参数，可以共享
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # 下采样的模块
        self.conv0_0 = DoubleConvBlock(in_channels, nb_filters[0] * 2, nb_filters[0] * 2) # 3 -> 32*2
        self.conv1_0 = DoubleConvBlock(nb_filters[0], nb_filters[1], nb_filters[1]) # 32 -> 64
        self.conv2_0 = DoubleConvBlock(nb_filters[1], nb_filters[2], nb_filters[2]) # 64 -> 128
        self.conv3_0 = DoubleConvBlock(nb_filters[2], nb_filters[3], nb_filters[3]) # 128 -> 256
        self.conv4_0 = DoubleConvBlock(nb_filters[3], nb_filters[4], nb_filters[4]) # 256 -> 512

        # 处理UNET和fpn模块图像之间的关联 1×1卷积
        self.conv0_1 = nn.Conv2d(nb_filters[0], nb_filters[0], kernel_size=1, stride=1)
        self.conv1_1 = nn.Conv2d(nb_filters[1], nb_filters[0], kernel_size=1, stride=1)
        self.conv2_1 = nn.Conv2d(nb_filters[2], nb_filters[0], kernel_size=1, stride=1)
        self.conv3_1 = nn.Conv2d(nb_filters[3], nb_filters[0], kernel_size=1, stride=1)
        self.conv4_1 = nn.Conv2d(nb_filters[4], nb_filters[0], kernel_size=1, stride=1)

        # 3×3卷积
        self.conv2 = nn.Conv2d(nb_filters[0], nb_filters[0], kernel_size=3, padding=1, stride=1)

        # bottleneck中加入ASPP模块
        self.aspp = ASPP(nb_filters[4], nb_filters[4], [6, 12, 18])
        self.aspp0 = ASPP(nb_filters[0], nb_filters[0], [6, 12, 18])

        # 上采样的模块
        self.conv3_3 = DoubleConvBlock(nb_filters[4] + nb_filters[0], nb_filters[3], nb_filters[3])  # 512+32 -> 256
        self.conv2_3 = DoubleConvBlock(nb_filters[3] + nb_filters[0], nb_filters[2], nb_filters[2])  # 256+32 -> 128
        self.conv1_3 = DoubleConvBlock(nb_filters[2] + nb_filters[0], nb_filters[1], nb_filters[1])  # 128+32 -> 64
        self.conv0_3 = DoubleConvBlock(nb_filters[1] + nb_filters[0], nb_filters[0], nb_filters[0])  # 64+32 -> 32

        # 最后接一个Conv计算在所有类别上的分数
        self.final = nn.Conv2d(nb_filters[0], num_classes, kernel_size=1, stride=1)  # 32 -> 2

    def forward(self, input):
        # 下采样编码
        x0_0, f0_0 =self.conv0_0(input).chunk(2, dim=1)

        x1_0 = self.conv1_0(self.down(x0_0))
        f1_0 = self.down(f0_0)

        x2_0 = self.conv2_0(self.down(x1_0))
        f2_0 = self.down(f1_0)

        x3_0 = self.conv3_0(self.down(x2_0))
        f3_0 = self.down(f2_0)

        x4_0 = self.conv4_0(self.down(x3_0))
        f4_0 = self.down(f3_0)

        if self.is_esa:
           x4_0 = x4_0 + self.is_ese_4_0(x4_0)
        if self.is_aspp:
           x4_0 = self.aspp(x4_0)

        f4_0 = f4_0 + self.conv4_1(x4_0)
        # 实现了特征金字塔模块
        # 自上而下的路径聚合模块和自下而上的路径聚合模块
        f5_0 = f3_0 + self.up(f4_0)
        f6_0 = f2_0 + self.up(f5_0)
        f7_0 = f1_0 + self.up(f6_0)
        f8_0 = self.up(f7_0)
        f9_0 = f7_0 + self.down(f8_0)
        f10_0 = f6_0 + self.down(f9_0)
        f11_0 = f5_0 + self.down(f10_0)

        if self.is_aspp:
            f8_0 =self.aspp0(f8_0)

        if self.is_esa:
            f8_0 = f8_0 + self.is_ese_0_0(f8_0)

        # 特征融合并进行上采样解码，使用concatenate进行特征融合
        x5_0 = self.conv3_3(torch.cat([self.conv2(f11_0), self.up(x4_0)], dim=1))
        x6_0 = self.conv2_3(torch.cat([self.conv2(f10_0), self.up(x5_0)], dim=1))
        x7_0 = self.conv1_3(torch.cat([self.conv2(f9_0), self.up(x6_0)], dim=1))
        x8_0 = self.conv0_3(torch.cat([self.conv2(f8_0), self.up(x7_0)], dim=1))

        # 计算每个类别上的得分
        output = self.final(x8_0)

        return output

if __name__ == '__main__':
    x=torch.rand([4, 3, 512, 512])
    net=UNet_fpn(num_classes=2, in_channels=3)
    print(net(x).shape)