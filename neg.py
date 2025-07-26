from PIL import Image
import numpy as np


def invert_image_colors(image_path, output_path):
    """
    读取PNG图片并黑白颠倒（颜色反相）

    参数：
        image_path: 输入图片路径
        output_path: 输出图片保存路径
    """
    # 打开图片
    img = Image.open(image_path)

    # 转换为RGB模式（确保处理彩色或灰度图）
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # 将图像转换为numpy数组
    img_array = np.array(img)

    # 颜色反相（黑白颠倒）
    inverted_array = 255 - img_array

    # 转换回图像
    inverted_img = Image.fromarray(inverted_array)

    # 保存结果
    inverted_img.save(output_path)
    print(f"处理完成，结果已保存至：{output_path}")


# 使用示例
input_path = "D:/1.png"  # 替换为你的输入图片路径
output_path = "D:/1_o.png"  # 输出图片路径
invert_image_colors(input_path, output_path)