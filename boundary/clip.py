import torch.nn as nn
import torch
# 模型定义
from transformers import CLIPModel, CLIPProcessor

# 加载 CLIP 模型和处理器
clip_model = CLIPModel.from_pretrained("/root/autodl-tmp/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("/root/autodl-tmp/clip-vit-base-patch32")

# 提取视觉编码器
vision_model = clip_model.vision_model
class SegmentationModel(nn.Module):
    def __init__(self, vision_model, num_classes):
        super().__init__()
        self.vision_model = vision_model
        self.decoder = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1),  # 假设视觉编码器输出通道为 768
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1),     # 输出类别数
            nn.Upsample(scale_factor=32, mode="bilinear", align_corners=False)  # 上采样到 224x224
        )

    def forward(self, x):
        # 提取视觉特征
        visual_features = self.vision_model(x).last_hidden_state  # [batch_size, 50, 768]
        
        # 移除 [CLS] token
        visual_features = visual_features[:, 1:, :]  # [batch_size, 49, 768]
        # 调整特征形状 [batch_size, 768, 7, 7]
        batch_size = visual_features.shape[0]
        visual_features = visual_features.permute(0, 2, 1).view(batch_size, 768, 7, 7)
        
        # 解码器生成分割掩码
        mask = self.decoder(visual_features)  # [batch_size, num_classes, 224, 224]
        return mask
# 初始化模型
num_classes = 2  # 假设是二分类任务
model = SegmentationModel(vision_model, num_classes).to("cuda")

# 加载本地权重
checkpoint = torch.load("/root/autodl-tmp/clip_segmentation_145_0.792_model.pth")
model.load_state_dict(checkpoint)

if __name__ == '__main__':
    x=torch.rand([4, 3, 224, 224]).to('cuda')
    # avg = nn.AvgPool2d(3)
    # print(avg(x).shape)
    print(model(x).shape)