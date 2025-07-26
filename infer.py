import torch
import torch.nn as nn
from segment_anything import sam_model_registry, SamPredictor
import argparse
from torch.nn import functional as F
from boundary.model import UNet_border
from boundary.border import border
import numpy as np
from boundary.LevelSet import TorchLevelSet
from utils.metrics import compute_metrics
from boundary.qkv import ImageMultiHeadAttention
from torchvision import transforms

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="workdir", help="work dir")
    parser.add_argument("--run_name", type=str, default="sam-med2d", help="run model name")
    parser.add_argument("--epochs", type=int, default=15, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="train batch size")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")
    parser.add_argument("--mask_num", type=int, default=5, help="get mask number")
    parser.add_argument("--data_path", type=str, default="data_demo", help="train data path")
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--resume", type=str, default=None, help="load resume")
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument("--sam_checkpoint", type=str, default="pretrain_model/epoch15_sam.pth", help="sam checkpoint")
    parser.add_argument("--iter_point", type=int, default=8, help="point iterations")
    parser.add_argument('--lr_scheduler', type=str, default=None, help='lr scheduler')
    parser.add_argument("--point_list", type=list, default=[1, 3, 5, 9], help="point_list")
    parser.add_argument("--multimask", type=bool, default=True, help="ouput multimask")
    parser.add_argument("--encoder_adapter", type=bool, default=True, help="use adapter")
    parser.add_argument("--use_amp", type=bool, default=False, help="use amp")
    args = parser.parse_args()
    if args.resume is not None:
        args.sam_checkpoint = None
    return args

# 无提示增强版
def prompt_and_decoder(args, prompt, model, image_embeddings, decoder_iter=False):

    if decoder_iter:
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=prompt.get('points'),
                boxes=prompt.get('boxes'),
                masks=prompt.get('masks'),
            )

    else:
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=prompt.get('points'),
            boxes=prompt.get('boxes'),
            masks=prompt.get('masks'),
        )

    low_res_masks, iou_predictions = model.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=args.multimask,
    )

    if args.multimask:
        max_values, max_indexs = torch.max(iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_predictions = max_values
        low_res = []
        for i, idx in enumerate(max_indexs):
            low_res.append(low_res_masks[i:i + 1, idx])
        low_res_masks = torch.stack(low_res, 0)

    masks = F.interpolate(low_res_masks, (args.image_size, args.image_size), mode="bilinear", align_corners=False, )
    return masks, low_res_masks, iou_predictions

#该函数用于生成掩码
def tomask(image):
    return torch.max(image, dim=1)[1]

class inferLLM(nn.Module):
    def __init__(self, args, prompt, device='cpu'):
        super().__init__()
        self.args = args
        self.device = device
        self.prompt = prompt
    def forward(self, image):
        model = sam_model_registry["vit_b"](self.args).to(self.device)
        # print(model.image_encoder.img_size)
        embedding = model.image_encoder(image)
        masks, _ , _ = prompt_and_decoder(self.args, self.prompt, model, embedding, decoder_iter=False)
        return masks


# 计算均值和标准差
def compute_stats(batch):
    # 展平所有图像的像素（维度：通道 × (批量×高×宽)）
    channels = batch.shape[1]  # 3
    flattened = batch.permute(1, 0, 2, 3).reshape(channels, -1)  # [3, 4*256*256]

    mean = flattened.mean(dim=1)  # 按通道求均值
    std = flattened.std(dim=1)  # 按通道求标准差
    return mean, std

class OurModel(nn.Module):
    def __init__(self, num_classes, in_channels, threshold = 0.8):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        # 定义一个1×3的可学习权重矩阵 [w1, w2, w3]
        self.weights = torch.nn.Parameter(torch.ones(1, 3))  # 初始化为[1.0, 1.0, 1.0]
        self.threshold = threshold
        self.pool = nn.AvgPool2d(kernel_size=4, stride=4)
        self.small = UNet_border(self.num_classes,self.in_channels)
        self.levelset = TorchLevelSet()
        self.trans = ImageMultiHeadAttention(embed_dim=3,num_heads=1)
        self.final = nn.Conv2d(in_channels=3,out_channels=1,kernel_size=3)

    def forward(self, image):
         # border = UNet_border(self.num_classes,self.in_channels)
         # 图像增强模块
         aug_image = transforms.Compose([
             transforms.Resize(256),
             transforms.RandomCrop(size=256,
                                   padding=32,
                                   padding_mode='reflect'),
             transforms.RandomHorizontalFlip(),
             # transforms.ToTensor(),
             transforms.Normalize(mean=compute_stats(image)[0],std=compute_stats(image)[1])
         ])
         # 逐图像处理
         processed_images = []
         for img_tensor in image:  # img_tensor: [3, 256, 256]
             # img_pil = transforms.ToPILImage()(img_tensor)  # 必须转为PIL图像
             transformed = aug_image(img_tensor)  # 应用变换
             processed_images.append(transformed)
         # 合并结果
         aug_res = torch.stack(processed_images, dim=0)  # [4, 3, 256, 256]

         res_border = tomask(self.small(aug_res)[0])
         prompt0 = {
             'points': None,
             'boxes': None,
             'masks': None
         }
         llm = inferLLM(args=parse_args(), prompt=prompt0)
         res_llm = tomask(llm(image))

         # 初始化水平集
         init_LSF = torch.zeros(4, 1, 256, 256) - 1  # 全-1初始化
         res_levelset = self.levelset.seg(init_LSF, image.mean(dim=1, keepdim=True), num_iters=100).squeeze(1)

         res_model = tomask(self.small(image)[0])

         if( compute_metrics(res_border, res_llm, 1)[1] <= self.threshold):
            res = self.weights[0,0] * res_border + self.weights[0,1] * res_llm + self.weights[0,2] * res_model
            #125行归一化成0-1，再做一个tomask
         else:
            prompt1 = {
               'points':None,
               'boxes':None,
               'masks':self.pool(self.small(image)[0])
            }
            llm_prompt = inferLLM(args=parse_args(),prompt=prompt1)
            res = self.weights[0,0] * res_border + self.weights[0,1] * tomask(llm_prompt(image)) + self.weights[0,2] * res_levelset
         guide_boundary = self.final(self.trans(torch.stack([self.weights[0,0] * res_border, self.weights[0,1] * res_llm, self.weights[0,2] * res_levelset], dim=1))).squeeze(1)
         #134行把mask之前的取出来用
         return res, self.small(image), res_llm, res_levelset, res_model, self.weights, guide_boundary

if __name__ == '__main__':
  # 使用大模型输出
  args = parse_args()
  x = torch.rand([4, 3, 256, 256])
  net = OurModel(num_classes=1, in_channels=3)
  res = net(x)[0]
  # print(border(res.detach().numpy().astype(np.int8))) # 生成边界
  print(res.shape)