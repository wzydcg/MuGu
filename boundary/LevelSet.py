import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

class TorchLevelSet:
    def __init__(self, mu=2, nu=0.005 * 255 * 255, epison=1, step=0.1,
                 lambda1=2, lambda2=1, kernel_size=5):
        self.mu = torch.nn.Parameter(torch.tensor(2.0))
        self.nu = torch.nn.Parameter(torch.tensor(0.005 * 255 * 255))
        self.epison = torch.nn.Parameter(torch.tensor(1.0))
        self.step = torch.nn.Parameter(torch.tensor(0.1))
        self.lambda1 = torch.nn.Parameter(torch.tensor(2.0))
        self.lambda2 = torch.nn.Parameter(torch.tensor(1.0))
        # 创建高斯核 (可学习参数)
        self.kernel = self._create_kernel(kernel_size)

    def _create_kernel(self, size):
        """创建可分离的高斯卷积核"""
        kernel = torch.ones((1, 1, size, size), dtype=torch.float32)
        kernel = kernel / (size ** 2)
        return kernel.to(device='cuda' if torch.cuda.is_available() else 'cpu')

    def _apply_filter(self, x, kernel):
        """PyTorch实现的2D卷积"""
        # x: [B,C,H,W], kernel: [1,1,K,K]
        return F.conv2d(x, kernel, padding=kernel.shape[-1] // 2, groups=x.size(1))

    def seg(self, LSF, img, num_iters=10):
        """
        输入:
            LSF: 初始水平集 [B,1,H,W]
            img: 输入图像 [B,C,H,W]
            num_iters: 迭代次数
        返回:
            分割结果 [B,1,H,W]
        """
        for _ in range(num_iters):
            LSF = self._rsf_step(LSF, img)
        return torch.sigmoid(LSF)  # 转换为概率图

    def _rsf_step(self, LSF, img):
        """单次RSF迭代"""
        # 1. 计算Heaviside函数及其导数
        LSF_prob = torch.sigmoid(LSF)
        Drc = (self.epison / math.pi) / (self.epison ** 2 + LSF ** 2)
        Hea = 0.5 * (1 + (2 / math.pi) * torch.atan(LSF / self.epison))

        # 2. 计算曲率项
        grad_x = F.conv2d(LSF, torch.tensor([[[[0, 0, 0], [-1, 0, 1], [0, 0, 0]]]],
                                            dtype=torch.float32, device=LSF.device), padding=1)
        grad_y = F.conv2d(LSF, torch.tensor([[[[0, -1, 0], [0, 0, 0], [0, 1, 0]]]],
                                            dtype=torch.float32, device=LSF.device), padding=1)

        s = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        Nx = grad_x / s
        Ny = grad_y / s

        # 3. 计算惩罚项
        lap_kernel = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]],
                                  dtype=torch.float32, device=LSF.device)
        Lap = F.conv2d(LSF, lap_kernel, padding=1)

        # 4. 计算区域项
        KIH = self._apply_filter(Hea * img, self.kernel)
        KH = self._apply_filter(Hea, self.kernel)
        f1 = KIH / (KH + 1e-8)

        KIH1 = self._apply_filter((1 - Hea) * img, self.kernel)
        KH1 = self._apply_filter(1 - Hea, self.kernel)
        f2 = KIH1 / (KH1 + 1e-8)

        R1 = (self.lambda1 - self.lambda2) * img ** 2
        R2 = self._apply_filter(self.lambda1 * f1 - self.lambda2 * f2, self.kernel)
        R3 = self._apply_filter(self.lambda1 * f1 ** 2 - self.lambda2 * f2 ** 2, self.kernel)
        RSFterm = -Drc * (R1 - 2 * R2 * img + R3)

        # 5. 更新水平集函数
        delta_LSF = self.step * (self.nu * Drc * Lap + self.mu * (Lap) + RSFterm)
        return LSF + delta_LSF


# 使用示例
if __name__ == "__main__":
    # 模拟输入 [B,C,H,W]
    img_tensor = torch.rand(4, 3, 256, 256)  # 假设已经归一化到[0,1]
    # 初始化水平集
    init_LSF = torch.zeros(4, 1, 256, 256) - 1  # 全-1初始化
    # 创建模型
    levelset = TorchLevelSet()
    # 前向计算
    result = levelset.forward(init_LSF, img_tensor.mean(dim=1, keepdim=True), num_iters=20).squeeze(1)
    print(result.shape)