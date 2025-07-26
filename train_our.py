#!/usr/bin/python3
# -*- coding: utf-8 -*
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from torch.utils.data import DataLoader
from models import unet,loss_function,unet_fpn
from boundary.model import UNet_border
from boundary.border import border
from datasets import mydataset
from torchvision import transforms
import os
import argparse
import numpy as np
from utils.metrics import compute_metrics
from torch import nn
import torch.nn.functional as F
from infer import OurModel

# 命令行参数设置
parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='LLM2', choices=['unet', 'unet_esa', 'our', 'unet_border','LLM2'])
parser.add_argument('--dataset', type=str, default='mydata', choices=['mydata'])
# parser.add_argument('--loss', type=str, default='Dice', choices=['Dice','CE'])
parser.add_argument('--optimizer', type=str, default='adam', choices=['adam'])
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--gpu', type=str, default='0', choices=['0', '1'])
parser.add_argument('--parallel', type=str, default='False', choices=['True', 'False'])
parser.add_argument('--num_workers', type=int, default=0, choices=list(range(17)))
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--lr', type=float, default=1e-4)   # 初始学习率， Adam
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--print_frequency', type=int, default=4)
parser.add_argument('--save_frequency', type=int, default=1)
args = parser.parse_args()

# 其他准备
BASE_PATH =r'/tmp/pycharm_project_264' if torch.cuda.is_available() else r'D:\Code\lungseg\trainingrecords'

format = '{}_{}_our_{}_{}'.format(args.dataset, args.model, args.optimizer, args.lr)


checkpoint_path_prefix = os.path.join(BASE_PATH, 'checkpoint', format)
os.makedirs(checkpoint_path_prefix, exist_ok=True)

#判断是否有gpu
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据
print('Loading data...')
# 选择数据集
if args.dataset == 'mydata':
    dataset = mydataset.Mydataset
else:
    print('数据集异常')
    pass

# 对image和mask转换为张量模式供后续训练
transform = transforms.Compose([transforms.ToTensor()])
target_transform = transforms.Compose([transforms.ToTensor()])

#BASE_PATH =r'/tmp/pycharm_project_135' if torch.cuda.is_available() else r'D:\Code\lungseg\trainingrecords'
if args.dataset == 'mydata':
    train_data = dataset(mode='train', transform=transform, target_transform=target_transform,
                         BASE_PATH= r"/tmp/pycharm_project_264/processed_data/train" if torch.cuda.is_available() else r"D:\Code\lungseg\processed_data\train")
    val_data = dataset(mode='val', transform=transform, target_transform=target_transform,
                       BASE_PATH=r"/tmp/pycharm_project_264/processed_data/val" if torch.cuda.is_available() else r"D:\Code\lungseg\processed_data\val")
else:
    pass


train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

print('Create model...')
# 选择网络模型
if args.model == 'unet':
    net = unet.UNet(num_classes=dataset.NUM_CLASSES, in_channels=dataset.CHANNELS_NUM,is_esa=False)
elif args.model == 'unet_esa':
    net = unet.UNet(num_classes=dataset.NUM_CLASSES, in_channels=dataset.CHANNELS_NUM, is_esa=True)
elif args.model == 'our':
    net =unet_fpn.UNet_improve(num_classes=dataset.NUM_CLASSES,in_channels=dataset.CHANNELS_NUM, is_esa=True ,is_aspp=True)
elif args.model == 'unet_border':
    net = UNet_border(num_classes=dataset.NUM_CLASSES,in_channels=dataset.CHANNELS_NUM)
elif args.model == 'LLM2':
    net = OurModel(num_classes=dataset.NUM_CLASSES,in_channels=dataset.CHANNELS_NUM)

# 设置优化方法和损失函数
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)


# 选择损失函数
criterion1 = torch.nn.CrossEntropyLoss()
criterion2 = loss_function.MultiClassDiceLoss(dataset.NUM_CLASSES)


print('<================== Parameters ==================>')
print('model: {}'.format(net))
print('dataset: {}(training={}, validation={})'.format(train_data, len(train_data), len(val_data)))
print('batch_size: {}'.format(args.batch_size))
print('batch_num: {}'.format(len(train_loader)))
print('epoch: {}'.format(args.epoch))
print('loss_function: {our}')
print('optimizer: {}'.format(optimizer))
print('<================================================>')

# 判断是否使用多GPU运行
if args.parallel == 'True':
    print('Use DataParallel.')
    net = torch.nn.DataParallel(net)


net = net.to(DEVICE)

start_epoch = 0
temp = 0
# 加载模型
if args.checkpoint is not None:
    checkpoint_data = torch.load(args.checkpoint)
    print('**** Load model and optimizer data from {} ****'.format(args.checkpoint))

    # 加载模型和优化器的数据
    net.load_state_dict(checkpoint_data['model_state_dict'])
    optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])

    # 加载上次训练的最后一个epoch和打印的最后一个temp，这里作为起点，需要在之前的基础上加1
    start_epoch = checkpoint_data['epoch'] + 1
    temp = checkpoint_data['temp'] + 1

    args.epoch += start_epoch

# 训练与验证的过程
print('Start training...')
# 设置早停初始值
best_dice = 0.0
for epoch in range(start_epoch, args.epoch):
    # 训练
    loss_all = []
    predictions_all = []
    labels_all = []
    print('-------------------------------------- Training {} --------------------------------------'.format(epoch + 1))
    net.train()
    for index, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        outputs, res_small, res_llm, res_model, _, guide_boundary = net(inputs)  # 如果是我们改进的模糊边界unet模型

        down = nn.MaxPool2d(kernel_size=2, stride=2)
        D1 = torch.tensor(border(guide_boundary.numpy())).float()
        D2 = down(D1)
        D3 = down(D2)
        D4 = down(D3)
        D5 = down(D4)

        optimizer.zero_grad()


        loss = 0
        # 如果使用deep supervision，返回1个list（包含多个输出），计算每个输出的loss，最后求平均
        if isinstance(outputs, list):
            for out in outputs:
                loss += criterion1(out, labels.long())
            loss /= len(outputs)
        else:
            loss = criterion1(outputs, labels.long()) + criterion2(outputs, labels.long()) + criterion1(res_small[1],D1.long()) + criterion1(res_small[2],D2.long()) + criterion1(res_small[3],D3.long()) + criterion1(res_small[4],D4.long()) + criterion1(res_small[5],D5.long()) + criterion1(res_small[6],D4.long()) + criterion1(res_small[7],D3.long()) + criterion1(res_small[8],D2.long()) + criterion1(res_small[9],D1.long())
            loss = loss + criterion1(res_small[0], labels.long()) + criterion1(res_model, 1 - labels.long())
            loss = loss + criterion1(res_levelset, res_small[0]) + criterion1(1 - res_levelset, res_model)
        # 计算在该批次上的平均损失函数
        loss /= inputs.size(0)

        # 更新网络参数
        loss.backward()
        optimizer.step()

        loss_all.append(loss.item())

        if isinstance(outputs, list):
            # 若使用deep supervision，用最后的输出来进行预测
            predictions = torch.max(outputs[-1], dim=1)[1].cpu().numpy().astype(np.int8)
        else:
            # 将概率最大的类别作为预测的类别
            predictions = torch.max(outputs, dim=1)[1].cpu().numpy().astype(np.int8)

        labels = labels.cpu().numpy().astype(np.int8)

        predictions_all.append(predictions)
        labels_all.append(labels)

        if (index + 1) % args.print_frequency == 0:
            # 计算打印间隔的平均损失函数
            avg_loss = np.mean(loss_all)
            loss_all = []

            temp += 1

            print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} ".format(
                epoch + 1, args.epoch, index + 1, len(train_loader), avg_loss))

    # 使用混淆矩阵计算语义分割中的指标
    iou, miou, dsc, mdsc, ac, pc, mpc, se, mse, sp, msp, f1, mf1 = compute_metrics(predictions_all, labels_all,
                                                                                   dataset.NUM_CLASSES)



    print('Training: MIoU: {:.4f}, MDSC: {:.4f}, MPC: {:.4f}, AC: {:.4f}, MSE: {:.4f}, MSP: {:.4f}, MF1: {:.4f}'.format(
        miou, mdsc, mpc, ac, mse, msp, mf1
    ))

    # 验证
    loss_all = []
    predictions_all = []
    labels_all = []

    print('-------------------------------------- Validation {} ------------------------------------'.format(epoch + 1))

    net.eval()
    with torch.no_grad():
        for _, data in enumerate(val_loader):
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs, res_small, res_llm, res_levelset, _, guide_boundary = net(inputs)
            down = nn.MaxPool2d(kernel_size=2, stride=2)
            D1 = torch.tensor(border(guide_boundary.numpy())).float()
            D2 = down(D1)
            D3 = down(D2)
            D4 = down(D3)
            D5 = down(D4)

            loss = 0
            # 如果使用deep supervision，返回1个list（包含多个输出），计算每个输出的loss，最后求平均
            if isinstance(outputs, list):
                for out in outputs:
                    loss += criterion1(out, labels.long())
                loss /= len(outputs)
            else:
                loss = criterion1(outputs, labels.long()) + criterion2(outputs, labels.long()) + criterion1(res_small[1],D1.long()) + criterion1(res_small[2],D2.long()) + criterion1(res_small[3],D3.long()) + criterion1(res_small[4],D4.long()) + criterion1(res_small[5],D5.long()) + criterion1(res_small[6],D4.long()) + criterion1(res_small[7],D3.long()) + criterion1(res_small[8],D2.long()) + criterion1(res_small[9],D1.long())
                loss = loss + criterion1(res_small[0], outputs) + criterion1(res_levelset, labels.long())
                loss = loss + criterion1(res_levelset, res_small[0]) + criterion1(1 - res_levelset, res_model)
            # 计算在该批次上的平均损失函数
            loss /= inputs.size(0)

            loss_all.append(loss.item())

            if isinstance(outputs, list):
                # 若使用deep supervision，用最后一个输出来进行预测
                predictions = torch.max(outputs[-1], dim=1)[1].cpu().numpy().astype(np.int8)
            else:
                # 将概率最大的类别作为预测的类别
                predictions = torch.max(outputs, dim=1)[1].cpu().numpy().astype(np.int8)
            labels = labels.cpu().numpy().astype(np.int8)

            predictions_all.append(predictions)
            labels_all.append(labels)

    # 使用混淆矩阵计算语义分割中的指标
    iou, miou, dsc, mdsc, ac, pc, mpc, se, mse, sp, msp, f1, mf1 = compute_metrics(predictions_all, labels_all,
                                                                                   dataset.NUM_CLASSES)
    avg_loss = np.mean(loss_all)

    # 绘制每个类别的IoU
    temp_dict = {'miou': miou}
    for i in range(dataset.NUM_CLASSES):
        temp_dict['class{}'.format(i)] = iou[i]

    print('Training: MIoU: {:.4f}, MDSC: {:.4f}, MPC: {:.4f}, AC: {:.4f}, MSE: {:.4f}, MSP: {:.4f}, MF1: {:.4f}'.format(
        miou, mdsc, mpc, ac, mse, msp, mf1
    ))

    # 保存模型参数和优化器参数
    if mdsc >= best_dice:
        best_dice = mdsc
        checkpoint_path = '{}_best_epoch_{}_{}.pkl'.format(format, str(epoch), str(np.round(mdsc, 3)))
        # save_checkpoint_path = checkpoint_path_prefix + '/' + checkpoint_path
        save_checkpoint_path = os.path.join(checkpoint_path_prefix, checkpoint_path)
        torch.save({
            'is_parallel': args.parallel,
            'epoch': epoch,
            'temp': temp,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()},
            save_checkpoint_path)
        print('Save model at {}.'.format(save_checkpoint_path))
