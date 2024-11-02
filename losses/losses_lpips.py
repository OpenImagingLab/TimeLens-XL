import pyiqa
from tools.registery import LOSS_REGISTRY
import torch
from torch.nn import functional as F


@LOSS_REGISTRY.register()
class lpips():
	def __init__(self, loss_dict):
		self.func = pyiqa.create_metric('lpips', as_loss=loss_dict.as_loss, device=torch.device("cuda"))
		self.weight = loss_dict.weight

	def forward(self, x, y):
		return self.func(x, y)*self.weight
    
@LOSS_REGISTRY.register()
class halflpips():
	def __init__(self, loss_dict):
		self.func = pyiqa.create_metric('lpips', as_loss=loss_dict.as_loss, device=torch.device("cuda"))
		self.weight = loss_dict.weight

	def forward(self, x, y):
		if len(x.shape) == 5:
			N, T, C, H, W = x.shape
			x, y = x.view(-1, C, H, W), y.view(-1, C, H, W)
		x, y = F.interpolate(x, scale_factor=0.5, mode='bilinear'), F.interpolate(y, scale_factor=0.5, mode='bilinear')
		return self.func(x, y)*self.weight