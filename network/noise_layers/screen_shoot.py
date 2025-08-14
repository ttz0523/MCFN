import math
import random

import kornia
import numpy as np
import torch
import torch.nn as nn
from kornia.geometry.transform import get_perspective_transform
from kornia.geometry.transform import warp_perspective
import time

def _compute_translation_matrix(translation: torch.Tensor) -> torch.Tensor:
    """Computes affine matrix for translation."""
    matrix: torch.Tensor = torch.eye(
        3, device=translation.device, dtype=translation.dtype)
    matrix = matrix.repeat(translation.shape[0], 1, 1)

    dx, dy = torch.chunk(translation, chunks=2, dim=-1)
    matrix[..., 0, 2:3] += dx
    matrix[..., 1, 2:3] += dy
    return matrix


def _compute_tensor_center(tensor: torch.Tensor) -> torch.Tensor:
    """Computes the center of tensor plane for (H, W), (C, H, W) and (B, C, H, W)."""
    assert 2 <= len(tensor.shape) <= 4, f"Must be a 3D tensor as HW, CHW and BCHW. Got {tensor.shape}."
    height, width = tensor.shape[-2:]
    center_x: float = float(width - 1) / 2
    center_y: float = float(height - 1) / 2
    center: torch.Tensor = torch.tensor(
        [center_x, center_y],
        device=tensor.device, dtype=tensor.dtype)
    return center


def _compute_scaling_matrix(scale: torch.Tensor,
                            center: torch.Tensor) -> torch.Tensor:
    """Computes affine matrix for scaling."""
    # angle: torch.Tensor = torch.zeros_like(scale)
    angle: torch.Tensor = torch.zeros(scale.shape[0])
    matrix: torch.Tensor = kornia.get_rotation_matrix2d(center, angle, scale)
    return matrix


def _compute_rotation_matrix(angle: torch.Tensor,
                             center: torch.Tensor) -> torch.Tensor:
    """Computes a pure affine rotation matrix."""
    scale: torch.Tensor = torch.ones((angle.shape[0], 2))
    matrix: torch.Tensor = kornia.get_rotation_matrix2d(center, angle, scale)
    return matrix


def translate(image, device, d=8):
    c = image.shape[0]
    h = image.shape[-2]
    w = image.shape[-1]  # destination size
    trans = torch.ones(c, 2)
    for i in range(c):
        dx = random.uniform(-d, d)
        dy = random.uniform(-d, d)

        trans[i, :] = torch.tensor([
            [dx, dy],
        ])
    translation_matrix: torch.Tensor = _compute_translation_matrix(trans)
    matrix = translation_matrix[..., :2, :3]
    # warping needs data in the shape of BCHW
    is_unbatched: bool = image.ndimension() == 3
    if is_unbatched:
        image = torch.unsqueeze(image, dim=0)

    # we enforce broadcasting since by default grid_sample it does not
    # give support for that
    matrix = matrix.expand(image.shape[0], -1, -1).to(device)

    # warp the input tensor
    data_warp: torch.Tensor = kornia.warp_affine(image, matrix, dsize=(h, w), padding_mode='border').to(device)

    # return in the original shape
    if is_unbatched:
        data_warp = torch.squeeze(data_warp, dim=0)

    return data_warp


def rotate(image, device, d=8):
    c = image.shape[0]
    h = image.shape[-2]
    w = image.shape[-1]  # destination size
    angle = torch.ones(c)
    center = torch.ones(c, 2)
    for i in range(c):
        # scale_factor
        an = random.uniform(-d, d)
        angle[i] = torch.tensor([an])
        # center
        center[i, :] = torch.tensor([[h / 2 - 1, w / 2 - 1], ])
    # compute the tensor center
    if center is None:
        center = _compute_tensor_center(image)

    angle = angle.expand(image.shape[0])
    center = center.expand(image.shape[0], -1)
    rotation_matrix: torch.Tensor = _compute_rotation_matrix(angle, center)

    # warp using the affine transform
    # affine(tensor, scaling_matrix[..., :2, :3], align_corners=align_corners)
    matrix = rotation_matrix[..., :2, :3]
    # warping needs data in the shape of BCHW
    is_unbatched: bool = image.ndimension() == 3
    if is_unbatched:
        image = torch.unsqueeze(image, dim=0)

    # we enforce broadcasting since by default grid_sample it does not
    # give support for that
    matrix = matrix.expand(image.shape[0], -1, -1).to(device)

    # warp the input tensor
    data_warp: torch.Tensor = kornia.warp_affine(image, matrix, dsize=(h, w), padding_mode='border').to(device)

    # return in the original shape
    if is_unbatched:
        data_warp = torch.squeeze(data_warp, dim=0)

    return data_warp


def perspective(image, device, d=8):
    # the source points are the region to crop corners
    c = image.shape[0]
    h = image.shape[2]
    w = image.shape[3]  # destination size
    image_size = h
    points_src = torch.ones(c, 4, 2)
    points_dst = torch.ones(c, 4, 2)
    for i in range(c):
        points_src[i, :, :] = torch.tensor([[
            [0., 0.], [w - 1., 0.], [w - 1., h - 1.], [0., h - 1.],
        ]])

        # the destination points are the image vertexes   
        # d=8   
        tl_x = random.uniform(-d, d)  # Top left corner, top
        tl_y = random.uniform(-d, d)  # Top left corner, left
        bl_x = random.uniform(-d, d)  # Bot left corner, bot
        bl_y = random.uniform(-d, d)  # Bot left corner, left
        tr_x = random.uniform(-d, d)  # Top right corner, top
        tr_y = random.uniform(-d, d)  # Top right corner, right
        br_x = random.uniform(-d, d)  # Bot right corner, bot
        br_y = random.uniform(-d, d)  # Bot right corner, right

        points_dst[i, :, :] = torch.tensor([[
            [tl_x, tl_y],
            [tr_x + image_size, tr_y],
            [br_x + image_size, br_y + image_size],
            [bl_x, bl_y + image_size],
        ]])
        # compute perspective transform
    M: torch.tensor = get_perspective_transform(points_src, points_dst).to(device)

    # warp the original image by the found transform
    data_warp: torch.tensor = warp_perspective(image.float(), M, dsize=(h, w)).to(device)

    return data_warp


def Light_Distortion_torch(c, embed_image):
    # 确保embed_image是torch.Tensor
    if not isinstance(embed_image, torch.Tensor):
        raise ValueError("embed_image must be a torch.Tensor")

    device = embed_image.device
    batch_size, channels, height, width = embed_image.size()

    mask_2d = torch.zeros(height, width, device=embed_image.device)
    # 计算a和b的值，确保它们是PyTorch张量
    a = (0.7 + torch.rand(1, device=device) * 0.2).item()
    b = (1.1 + torch.rand(1, device=device) * 0.2).item()
    # 根据c的值选择不同的处理方式
    if c == 0:
        # 初始化二维掩码
        for i in range(embed_image.shape[2]):
            mask_2d[i, :] = -((b - a) / (height - 1)) * (i - width) + a
        # 生成随机方向
        direction = torch.randint(1, 5, (1,), device=device).item()
        # 根据方向旋转二维掩码
        if direction == 1:
            O = mask_2d
        elif direction == 2:
            O = torch.rot90(mask_2d, 1, (0, 1))
        elif direction == 3:
            O = torch.rot90(mask_2d, 2, (0, 1))
        elif direction == 4:
            O = torch.rot90(mask_2d, 3, (0, 1))

    else:
        # 生成随机点
        x = torch.randint(0, width, (1,), device=device)
        y = torch.randint(0, height, (1,), device=device)

        # 计算最大长度
        max_len = max(torch.sqrt(x ** 2 + y ** 2), torch.sqrt((x - 255) ** 2 + y ** 2),
                      torch.sqrt(x ** 2 + (y - 255) ** 2),
                      torch.sqrt((x - width) ** 2 + (y - height) ** 2)
                      ).item()

        # 创建坐标网格
        x_coords = torch.arange(width, device=device).float()
        y_coords = torch.arange(height, device=device).float()
        X, Y = torch.meshgrid(x_coords, y_coords, indexing='ij')

        # 计算距离并应用光照扭曲效果
        distances = torch.sqrt((X - x) ** 2 + (Y - y) ** 2)
        mask = (distances / max_len * (a - b) + b).unsqueeze(0).unsqueeze(0).expand(batch_size, channels, height,
                                                                                    width).to(device)

        O = mask

    return O


def MoireGen_torch(p_size, theta, center_x, center_y):
    # 确保输入参数是torch.Tensor类型，并且它们具有正确的形状
    theta = theta.to(torch.float32)  # 转换为浮点数以进行数学运算
    center_x = center_x.to(torch.float32)
    center_y = center_y.to(torch.float32)

    # 创建一个与输出图像等大小的零张量
    M = torch.zeros((p_size, p_size), dtype=torch.float32)

    # 创建网格坐标
    x_coords = torch.arange(p_size, dtype=torch.float32) + 1  # 像素坐标从1开始
    y_coords = torch.arange(p_size, dtype=torch.float32) + 1
    x_grid, y_grid = torch.meshgrid(x_coords, y_coords)  # 生成网格坐标矩阵

    # 将网格坐标转换为torch.Tensor
    x_grid = x_grid.to(theta.device)
    y_grid = y_grid.to(theta.device)

    # 计算z1
    # 注意：这里我们假设center_x和center_y是单个值或具有广播能力的张量
    # 我们需要确保它们与x_grid和y_grid具有相同的形状
    if center_x.dim() == 0:
        center_x = center_x.unsqueeze(0).unsqueeze(0).expand_as(x_grid)
    if center_y.dim() == 0:
        center_y = center_y.unsqueeze(0).unsqueeze(0).expand_as(y_grid)

    dist_from_center = torch.sqrt((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2)
    z1 = 0.5 + 0.5 * torch.cos(2 * math.pi * dist_from_center)

    # 计算z2
    # 将theta转换为弧度
    theta_rad = theta / 180 * math.pi
    cos_theta = torch.cos(theta_rad)
    sin_theta = torch.sin(theta_rad)

    # 注意：这里我们假设theta是单个值或具有广播能力的张量
    # 我们需要确保cos_theta和sin_theta与x_grid和y_grid具有相同的形状
    if theta.dim() == 0:
        cos_theta = cos_theta.unsqueeze(0).unsqueeze(0).expand_as(x_grid)
        sin_theta = sin_theta.unsqueeze(0).unsqueeze(0).expand_as(y_grid)

    z2_x = cos_theta * x_grid
    z2_y = sin_theta * y_grid
    z2 = 0.5 + 0.5 * torch.cos(2 * math.pi * (z2_x + z2_y))

    # 取z1和z2中的最小值
    M = torch.min(z1, z2)

    # 将M的值域调整到[0, 1]
    M = (M + 1) / 2

    return M


def Moire_Distortion_torch(embed_image):
    # 确保embed_image是torch.Tensor，并且是在正确的设备上
    if not isinstance(embed_image, torch.Tensor):
        raise ValueError("embed_image must be a torch.Tensor")

    device = embed_image.device
    Z = torch.zeros(embed_image.size(0), 3, embed_image.size(2), embed_image.size(3), device=device)

    for i in range(3):
        theta = torch.randint(0, 180, (1,), device=device)
        center_x = torch.rand(1, device=device) * embed_image.size(2)
        center_y = torch.rand(1, device=device) * embed_image.size(3)

        # 注意：这里假设MoireGen_torch已经修改为接受PyTorch张量
        M = MoireGen_torch(embed_image.size(2), theta, center_x, center_y)

        # 确保M与Z的维度兼容
        if M.dim() != 2:
            raise ValueError("MoireGen_torch must return a 2D tensor")

        # 假设M是高度x高度的2D张量，我们需要将其扩展到与embed_image的通道维度兼容
        M = M.unsqueeze(0).unsqueeze(0).expand(1, 1, M.size(0), M.size(0)).to(device)

        Z[:, i, :, :] = M

    return Z


class ScreenShooting(nn.Module):

    def __init__(self, distance=2):
        super(ScreenShooting, self).__init__()
        self.distance = distance  # distance表示透视变化的扭曲程度

    def forward(self, image_and_cover):
        image, *_ = image_and_cover
        embed_image = image

        device = embed_image.device

        # perspective transform
        noised_image = perspective(embed_image, device, self.distance)

        # Light Distortion
        c = torch.randint(0, 2, (1,), device=device)
        L = Light_Distortion_torch(c, embed_image)

        # Moire Distortion
        Z = Moire_Distortion_torch(embed_image) * 2 - 1

        noised_image = noised_image * L * 0.85 + Z * 0.15

        # Gaussian noise
        noised_image = noised_image + 0.001 ** 0.5 * torch.randn(noised_image.size(), dtype=torch.float).to(device)

        return noised_image
