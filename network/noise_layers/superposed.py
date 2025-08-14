from . import Identity
import torch.nn as nn


class Superposed(nn.Module):

	def __init__(self, list=None):
		super(Superposed, self).__init__()
		if list is None:
			self.list_len = 1
			superposed_list = [Identity()]
		else:
			self.list_len = len(list)
			superposed_list = nn.Sequential(*list)  # *list表示将其中的元素按顺序解开成独立的元素作为形参

		self.list = superposed_list

	def forward(self, image_and_cover):
		noised_image, cover_image = image_and_cover
		for i in range(self.list_len):
			noised_image = self.list[i]([noised_image, cover_image])
		return noised_image

