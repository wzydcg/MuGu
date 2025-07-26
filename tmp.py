import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from boundary.model import UNet_border
from boundary.dual_unet import dual_UNet
import cv2
import matplotlib.pyplot as plt
# 1. 创建空模型（结构必须与训练时完全一致）
model = dual_UNet(in_channels=1,num_classes=2)
# 加载参数
# 直接使用 torch.load() 而非 pickle
state_dict = torch.load('G:/weight/polyp/dual_unet/769.pkl', map_location='cpu')  # 自动处理 persistent ID

model.load_state_dict(state_dict['model_state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.Resize((512, 512)),  # 与训练尺寸一致
    transforms.ToTensor()
])


def visualize_prediction(image_path, pred_mask):
    # 加载原始图像
    original_img = Image.open(image_path).convert("L")  # 保持灰度模式
    original_img = original_img.resize((512, 512))  # 确保尺寸匹配

    # 创建画布
    plt.figure(figsize=(15, 5))

    # 子图1：原始图像
    plt.subplot(1, 2, 1)
    plt.imshow(original_img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # 子图2：预测掩膜
    plt.subplot(1, 2, 2)
    plt.imshow(pred_mask, cmap='gray')
    plt.title('Prediction Mask')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('result_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def infer(image_path):
    img = Image.open(image_path).convert("L")
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(x)[0]
        pred = torch.sigmoid(output) > 0.5  # 二值化

    return pred.squeeze().numpy().astype(np.uint8) * 255  # 转为0-255的numpy数组

if __name__ == '__main__':
    image_path = 'G:/kvasir-seg/Kvasir-SEG/image/cju0roawvklrq0799vmjorwfv.png'
    pred = infer(image_path)
    pred = pred[1, :, :]
    # # 查找轮廓
    # contours, _ = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # # 计算周长（使用cv2.arcLength）
    # perimeter = cv2.arcLength(contours[2], closed=True)  # 假设只有一个连通域
    # print(f"Mask周长 (OpenCV): {perimeter:.2f} 像素")
    visualize_prediction(image_path,pred)
