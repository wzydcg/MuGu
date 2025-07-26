#!/usr/bin/python3
# -*- coding: utf-8 -*
# 所用到的所有网络的基本模块
from torch import nn
from boundary.Condconv import CondConv

class SingleConvBlock(nn.Module):
    """Conv-BN-ReLU的基本模块，其中的Conv并不会改变张量的尺寸"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.block(input)


class DoubleConvBlock(nn.Module):
    """由2个Conv-BN-ReLu组成的基本模块"""

    def __init__(self, in_channels, middle_channels, out_channels, dropout=False, p=0.2):
        super().__init__()
        # layers = [
        #     SingleConvBlock(in_channels, middle_channels),
        #     SingleConvBlock(middle_channels, out_channels)
        # ]
        #
        # if dropout:
        #     layers.append(nn.Dropout())

        # 将Dropout加在2个卷积层的中间
        layers = [SingleConvBlock(in_channels, middle_channels)]
        if dropout:
            layers.append(nn.Dropout(p=p))
        layers.append(SingleConvBlock(middle_channels, out_channels))

        self.block = nn.Sequential(*layers)

    def forward(self, input):
        return self.block(input)

