import json
import numpy as np
from tqdm import tqdm
import os
import shutil
from PIL import Image
import cv2


def mkdir():
    root = 'data_demo'
    if os.path.exists(root):
        shutil.rmtree(root)
    os.mkdir(root)
    os.mkdir(os.path.join(root, 'images'))
    os.mkdir(os.path.join(root, 'masks'))


# 生成训练集
def gen_trainSet(img_suff, msk_suff):
    p = 'RawData/train/images'
    image_list = [os.path.join(p, i) for i in os.listdir(p)]

    with open('data_demo/image2label_train.json', 'a') as jf:
        json_all = {}  # json文件
        for i in tqdm(image_list, desc='generate train set'):
            j = i.replace('images', 'masks').replace(img_suff, msk_suff)
            assert os.path.exists(j)  # 判断label是否存在

            shutil.copy(i, 'data_demo/images')

            mask = np.array(Image.open(j).convert('L'))  # 标签图像
            gray_list = np.unique(mask)

            img_list = []
            for gray in gray_list[1:]:  # 遍历mask所有的分割前景
                ret_mask = np.zeros(mask.shape, dtype=np.uint8)

                ret_mask[mask == gray] = 255  # 指定前景为255，其余为背景
                ret_mask[ret_mask < 255] = 0

                # 去除小的分割区域
                h, w = ret_mask.shape
                total_pixel = h * w
                if (np.sum(ret_mask != 0) / total_pixel) < 0.005:
                    continue

                ret_name = i.replace(img_suff, '_' + str(gray) + img_suff).replace('RawData/train/images',
                                                                                   'data_demo/masks')
                cv2.imwrite(ret_name, ret_mask)  # 保存生成的数据

                img_list.append(ret_name)
            if len(img_list) == 0:
                continue
            json_all[i.replace('RawData/train/images', 'data_demo/images')] = img_list

        json_str = json.dumps(json_all, indent=4)
        jf.write(json_str)


# 生成测试集
def gen_testSet(img_suff, msk_suff):
    p = 'RawData/test/images'
    image_list = [os.path.join(p, i) for i in os.listdir(p)]

    with open('data_demo/label2image_test.json', 'a') as jf:
        json_all = {}  # json文件
        for i in tqdm(image_list, desc='generate test set'):
            j = i.replace('images', 'masks').replace(img_suff, msk_suff)
            assert os.path.exists(j)  # 判断label是否存在

            shutil.copy(i, 'data_demo/images')

            mask = np.array(Image.open(j).convert('L'))  # 标签图像
            gray_list = np.unique(mask)

            for gray in gray_list[1:]:  # 遍历mask所有的分割前景
                ret_mask = np.zeros(mask.shape, dtype=np.uint8)

                ret_mask[mask == gray] = 255  # 指定前景为255，其余为背景
                ret_mask[ret_mask < 255] = 0

                # 去除小的分割区域
                h, w = ret_mask.shape
                total_pixel = h * w
                if (np.sum(ret_mask != 0) / total_pixel) < 0.005:
                    continue

                ret_name = i.replace(img_suff, '_' + str(gray) + img_suff).replace('RawData/test/images',
                                                                                   'data_demo/masks')
                cv2.imwrite(ret_name, ret_mask)  # 保存生成的数据

                json_all[ret_name] = i.replace('RawData/test/images', 'data_demo/images')

        json_str = json.dumps(json_all, indent=4)
        jf.write(json_str)


if __name__ == '__main__':
    imgFormat = '.png'  # image 的后缀
    maskFormat = '.png'  # mask 的后缀

    mkdir()  # 生成目录

    gen_trainSet(img_suff=imgFormat, msk_suff=maskFormat)  # 生成训练数据

    gen_testSet(img_suff=imgFormat, msk_suff=maskFormat)  # 生成测试数据
