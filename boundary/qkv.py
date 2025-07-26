import torch
import torch.nn as nn

class ImageMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(ImageMultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # Q, K, V 的线性映射
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        # 输出映射层
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = self.head_dim ** 0.5

    def forward(self, x):
        # 输入 x: (B, C, H, W)，需要 reshape 为 (B, N, C)
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)  # (B, N, C)

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        # 拆成多头 (B, num_heads, N, head_dim)
        Q = Q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # 注意力分数计算
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_probs, V)
        # 合并多头
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, H * W, self.embed_dim)
        # 输出映射
        out = self.out_proj(attn_out)
        # 恢复回原图维度 (B, C, H, W)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out

# # 测试示例
# # 假设输入是一张 14x14 的特征图（类似 patch embedding 后）
# img = torch.randn(4, 16, 16, 16)  # (B, C, H, W)
#
# mha = ImageMultiHeadAttention(embed_dim=16, num_heads=8)
# out = mha(img)
#
# print(out.shape)  # 输出应为 (4, 16, 16, 16)

