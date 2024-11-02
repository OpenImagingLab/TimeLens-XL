from .DISTS.DISTS_pytorch import DISTS
from tools.registery import LOSS_REGISTRY
import torch


@LOSS_REGISTRY.register()
class dists():
	def __init__(self, loss_dict):
		self.func = DISTS().to(torch.device("cuda"))
		self.weight = loss_dict.weight
		self.as_loss = loss_dict.as_loss

	def forward(self, x, y):
		if self.as_loss:
			return self.func(x, y)
		else:
			return self.func(x, y).detach()
