import torch
import torch.nn as nn
from kornia.filters import GaussianBlur2d


class Bright(nn.Module):

    def RGBAlgorithm(self, rgb_img, value=0.5, basedOnCurrentValue=True):
        img = rgb_img * 1.0
        # clamp and convert from [-1,1] to [0,255]
        img_out = (img.clamp(-1, 1) + 1) * 255 / 2
        img = img_out
        # 基于当前RGB进行调整（RGB*alpha）
        if basedOnCurrentValue:
            # 增量大于0，指数调整
            if value >= 0:
                alpha = 1 - value
                alpha = 1 / alpha

            # 增量小于0，线性调整
            else:
                alpha = value + 1

            img_out[:, 0, :, :] = img[:, 0, :, :] * alpha
            img_out[:, 1, :, :] = img[:, 1, :, :] * alpha
            img_out[:, 2, :, :] = img[:, 2, :, :] * alpha

        # 独立于当前RGB进行调整（RGB+alpha*255）
        else:
            alpha = value
            img_out[:, 0:1, :, :] = img[:, 0:1, :, :] + 255.0 * alpha
            img_out[:, 1:2, :, :] = img[:, 1:2, :, :] + 255.0 * alpha
            img_out[:, 2:3, :, :] = img[:, 2:3, :, :] + 255.0 * alpha

        img_out = img_out / 255.0

        # RGB颜色上下限处理(小于0取0，大于1取1)
        mask_3 = img_out < 0
        mask_4 = img_out > 1
        img_out = img_out * (~mask_3 + 0)
        img_out = img_out * (~mask_4 + 0) + mask_4

        # clamp and convert from [0,1] to [-1,1]
        img_out = img_out * 2 - 1
        return img_out

    def __init__(self, Increment, add_gf=False, gf_sigma=0):
        super(Bright, self).__init__()
        self.value = Increment / 255.0  # 范围-1至1
        self.basedOnCurrentValue = True  # 0或者1
        if gf_sigma is not True:
            self.add_gf = add_gf
            self.gf_sigma = gf_sigma
            self.gaussian_filter = GaussianBlur2d((3, 3), (gf_sigma, gf_sigma))

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover

        image = self.RGBAlgorithm(image, self.value, self.basedOnCurrentValue)
        if self.add_gf is True:
            image = self.gaussian_filter(image)
        return image
