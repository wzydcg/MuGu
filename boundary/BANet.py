import torch
import torch.nn as nn
import torch.nn.functional as F
from models.network_blocks import DoubleConvBlock
from models.esa_modules import ESA_block

class PEE(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes):
        super(PEE, self).__init__()
        # 1x1 卷积用于降维
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        # 保存池化尺寸
        self.pool_sizes = pool_sizes
        self.conv = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1, stride=1, padding=0)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Doubleconv = DoubleConvBlock(out_channels, out_channels, out_channels // 2)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x_reduced = self.conv1x1(x)  # 形状: (batch_size, out_channels, height, width)
        # Step 2: 多尺度平均池化
        edge_features = []
        for s in self.pool_sizes:
            # 平均池化
            avg_pool = F.avg_pool2d(x_reduced, kernel_size=s, stride=1, padding= s // 2)
            # Step 3: 边缘特征提取（特征图减去平均池化结果）
            edge_feature = x_reduced - avg_pool
            edge_features.append(edge_feature)
        edge_features.append(x_reduced)
        # 将多粒度边缘特征拼接在一起
        edge_features = self.conv(torch.cat(edge_features, dim=1))  # 形状: (batch_size, out_channels * S, height, width)
        
        fe = self.up(self.Doubleconv(edge_features))
        fs = self.up(self.Doubleconv(edge_features))
        # 这是交互注意力机制
        fe = fe + (1 - self.sig(fe)) * fs
        fs = fs + (1 - self.sig(fs)) * fe
        fm = torch.cat([fe,fs],dim=1)
        return self.down(fm)

# 示例用法
if __name__ == "__main__":
    # 输入特征图 (batch_size=1, in_channels=64, height=128, width=128)
    input_feature = torch.randn(4, 32, 512, 512)
    # 初始化模块 (in_channels=64, out_channels=32, pool_sizes=[2, 4, 8])
    edge_feature_extractor = PEE(in_channels=32, out_channels=32, pool_sizes=[3, 5])
    # 提取多粒度边缘特征
    edge_features = edge_feature_extractor(input_feature)
    print("输入特征图形状:", input_feature.shape)
    print("边缘特征图形状:", edge_features.shape)