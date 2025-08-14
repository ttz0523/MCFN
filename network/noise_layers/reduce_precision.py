import torch
import torch.nn as nn
import numpy as np


class RP(nn.Module):

    def __init__(self, npp):
        super(RP, self).__init__()
        self.npp = npp

    def reduce_precision_py(self, image, npp):
        """
        Reduce the precision of image, the numpy version.
        :param x: a float tensor, which has been scaled to [0, 1].
        :param npp: number of possible values per pixel. E.g. it's 256 for 8-bit gray-scale image, and 2 for binarized image.
        :return: a tensor representing image(s) with lower precision.
        """
        # Note: 0 is a possible value too.
        npp_int = npp - 1

        # 假设x是一个在GPU上的张量
        image = image.to('cuda')  # 将x移动到GPU上

        # 将x转换为NumPy数组
        image = image.cpu().numpy()

        x_int = np.rint(image * npp_int)  # 四舍五入

        x_float = x_int / npp_int
        return torch.from_numpy(x_float).cuda()

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover
        image = self.reduce_precision_py(image, self.npp)
        return image
