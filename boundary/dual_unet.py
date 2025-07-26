#!/usr/bin/python3
# -*- coding: utf-8 -*

# 这是我们的双任务unet
import torch
from torch import nn
from models.network_blocks import DoubleConvBlock
from boundary.SA import ImageMultiHeadAttention


class dual_UNet(nn.Module):

    def __init__(self, num_classes, in_channels):
        super().__init__()
        nb_filters = [32, 64, 128, 256, 512]
        # 标准的2倍上采样和下采样，因为没有可以学习的参数，可以共享
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # 下采样的模块
        self.conv0_0 = DoubleConvBlock(in_channels, nb_filters[0], nb_filters[0] * 2)
        self.conv1_0 = DoubleConvBlock(nb_filters[0], nb_filters[1], nb_filters[1])
        self.conv2_0 = DoubleConvBlock(nb_filters[1], nb_filters[2], nb_filters[2])
        self.conv3_0 = DoubleConvBlock(nb_filters[2], nb_filters[3], nb_filters[3])
        self.conv4_0 = DoubleConvBlock(nb_filters[3], nb_filters[4], nb_filters[4])

        # 分割分支上采样的模块
        self.conv3_1 = DoubleConvBlock(nb_filters[4] + nb_filters[3], nb_filters[3], nb_filters[3])
        self.conv2_2 = DoubleConvBlock(nb_filters[3] + nb_filters[2], nb_filters[2], nb_filters[2])
        self.conv1_3 = DoubleConvBlock(nb_filters[2] + nb_filters[1], nb_filters[1], nb_filters[1])
        self.conv0_4 = DoubleConvBlock(nb_filters[1] + nb_filters[0], nb_filters[0], nb_filters[0])

        self.pred = nn.Sequential(
            nn.Conv2d(nb_filters[4], nb_filters[4], kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(nb_filters[4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(nb_filters[4], nb_filters[4] * 2, kernel_size=3, padding=1, stride=1)
        )
        self.sa = ImageMultiHeadAttention(embed_dim=512, num_heads=8)
        # 最后接一个Conv计算在所有类别上的分数
        self.final = nn.Conv2d(nb_filters[0], num_classes, kernel_size=1, stride=1)

    def forward(self, input):
        # 下采样编码
        x = self.conv0_0(input)
        x0_0, y0_0 = x.chunk(2, dim=1)

        x1_0 = self.conv1_0(self.down(x0_0))
        y1_0 = self.conv1_0(self.down(y0_0))
        x2_0 = self.conv2_0(self.down(x1_0))
        y2_0 = self.conv2_0(self.down(y1_0))
        x3_0 = self.conv3_0(self.down(x2_0))
        y3_0 = self.conv3_0(self.down(y2_0))
        x4_0 = self.conv4_0(self.down(x3_0))
        y4_0 = self.conv4_0(self.down(y3_0))
        # 在bottleneck部分实现了一个交叉自注意力机制
        k1, v1 = self.pred(x4_0).chunk(2, dim=1)
        k2, v2 = self.pred(y4_0).chunk(2, dim=1)
        A1 = self.sa(x4_0, k1, v2)
        A2 = self.sa(x4_0, k2, v1)
        A3 = self.sa(x4_0, k1, v1)
        B1 = self.sa(y4_0, k1, v2)
        B2 = self.sa(y4_0, k2, v1)
        B3 = self.sa(y4_0, k2, v2)
        x4_0_1 = self.sa(A3, A1, A2)
        y4_0_1 = self.sa(B3, B1, B2)
        # 特征融合并进行上采样解码，使用concatenate进行特征融合
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0_1)], dim=1))
        y3_1 = self.conv3_1(torch.cat([x3_0, self.up(y4_0_1)], dim=1))

        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], dim=1))
        y2_2 = self.conv2_2(torch.cat([x2_0, self.up(y3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], dim=1))
        y1_3 = self.conv1_3(torch.cat([x1_0, self.up(y2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], dim=1))
        y0_4 = self.conv0_4(torch.cat([x0_0, self.up(y1_3)], dim=1))
        # 计算每个类别上的得分
        output1 = self.final(x0_4)
        output2 = self.final(y0_4)
        return output1, output2


if __name__ == '__main__':
    x = torch.rand([4, 3, 512, 512])
    net = dual_UNet(num_classes=2, in_channels=3)
    print(net(x)[0].shape)
    print(net(x)[1].shape)

