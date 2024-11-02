import torch
from torch.nn import Module
from torch.nn import functional as F
from torch import nn


class Correlation(Module):
	def __init__(self, pad_size=0, kernel_size=0, max_displacement=0, stride1=1, stride2=2, corr_multiply=1):
		super().__init__()
		self.pad_size = pad_size
		self.kernel_size = kernel_size
		self.max_displacement = max_displacement
		self.stride1 = stride1
		self.stride2 = stride2
		self.corr_multiply = corr_multiply
		self.sum_down = nn.AvgPool2d(kernel_size, stride=(stride1, stride2))

	def forward(self, input1, input2):
		input2_ = F.pad(input2, (self.pad_size, self.pad_size, self.pad_size, self.pad_size), mode='reflect')
		corr = []
		c, h1, w1 = input1.shape[1:]
		for hi in range(self.pad_size*2+1):
			for wi in range(self.pad_size*2+1):
				corr_cur_compute = input1*input2_[..., hi:hi+h1, wi:wi+w1]
				corr_cur_kerneldown = self.sum_down(corr_cur_compute)
				corr_cur = torch.sum(corr_cur_kerneldown, 1, keepdim=True)/(c)
				corr.append(corr_cur)
		return torch.cat(corr, 1)