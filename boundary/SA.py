import torch
import torch.nn as nn

import torch
import torch.nn as nn


class ImageMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(ImageMultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # 输出映射层
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = self.head_dim ** -0.5  # 注意这里应该是负的0.5次方

    def forward(self, q, k, v):
        """
        输入:
            q: (B, C, H, W)
            k: (B, C, H, W)
            v: (B, C, H, W)
        输出:
            out: (B, C, H, W)
        """
        B, C, H, W = q.shape

        # 将Q,K,V reshape为(B, N, C)
        q = q.view(B, C, H * W).permute(0, 2, 1)  # (B, N, C)
        k = k.view(B, C, H * W).permute(0, 2, 1)  # (B, N, C)
        v = v.view(B, C, H * W).permute(0, 2, 1)  # (B, N, C)

        # 拆成多头 (B, num_heads, N, head_dim)
        q = q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力分数计算
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_probs, v)

        # 合并多头
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, H * W, self.embed_dim)

        # 输出映射
        out = self.out_proj(attn_out)

        # 恢复回原图维度 (B, C, H, W)
        out = out.permute(0, 2, 1).view(B, C, H, W)

        return out

# # 测试示例
# # 假设输入是一张 14x14 的特征图（类似 patch embedding 后）
# img = torch.randn(4, 512, 32, 32)  # (B, C, H, W)

# mha = ImageMultiHeadAttention(embed_dim=512, num_heads=8)
# out = mha(img,img,img)

# # print(out.shape)  # 输出应为 (4, 64, 14, 14)

