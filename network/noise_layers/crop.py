import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def get_random_rectangle_inside(image_shape, height_ratio, width_ratio):
    image_height = image_shape[2]
    image_width = image_shape[3]

    remaining_height = int(height_ratio * image_height)
    remaining_width = int(width_ratio * image_width)

    if remaining_height == image_height:
        height_start = 0
    else:
        height_start = np.random.randint(0, image_height - remaining_height)

    if remaining_width == image_width:
        width_start = 0
    else:
        width_start = np.random.randint(0, image_width - remaining_width)

    return height_start, height_start + remaining_height, width_start, width_start + remaining_width


class Crop(nn.Module):

    def __init__(self, height_ratio, width_ratio):
        super(Crop, self).__init__()
        self.height_ratio = height_ratio
        self.width_ratio = width_ratio

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover

        h_start, h_end, w_start, w_end = get_random_rectangle_inside(image.shape, self.height_ratio,
                                                                     self.width_ratio)
        mask = torch.zeros_like(image)
        mask[:, :, h_start: h_end, w_start: w_end] = 1

        return image * mask


class Cropout(nn.Module):

    def __init__(self, height_ratio, width_ratio):
        super(Cropout, self).__init__()
        self.height_ratio = height_ratio
        self.width_ratio = width_ratio

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover
        cropout_mask = torch.zeros_like(image)
        h_start, h_end, w_start, w_end = get_random_rectangle_inside(image.shape, self.height_ratio,
                                                                     self.width_ratio)
        # cover_image[:, :, h_start: h_end, w_start: w_end] = image[:, :, h_start: h_end, w_start: w_end]
        cropout_mask[:, :, h_start:h_end, w_start:w_end] = 1

        cover_image = image * cropout_mask + cover_image * (1 - cropout_mask)
        return cover_image


class ResizeCrop(nn.Module):

    def __init__(self, height_ratio, width_ratio):
        super(ResizeCrop, self).__init__()
        self.height_ratio = height_ratio
        self.width_ratio = width_ratio

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover

        # 定义一个RandomResizedCrop变换
        transform = transforms.RandomResizedCrop((cover_image.shape[-2], cover_image.shape[-1]), scale=(self.height_ratio, self.width_ratio),
                                                 ratio=(1.0, 1.0))

        # 对图片进行变换
        img_transformed = transform(image)

        return img_transformed


class Dropout(nn.Module):

    def __init__(self, prob):
        super(Dropout, self).__init__()
        self.prob = prob

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover

        rdn = torch.rand(image.shape).to(image.device)
        output = torch.where(rdn > self.prob * 1., cover_image, image)
        return output
