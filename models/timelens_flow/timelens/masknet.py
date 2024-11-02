import torch
from torch import nn
import numpy as np
from torch.nn import functional as F


class ResBlock(nn.Module):
	def __init__(self, in_channel, base_channel):
		super().__init__()
		self.conv_in = nn.Sequential(
			nn.Conv2d(in_channel, base_channel, 3, 1, 1, bias=True),
			nn.LeakyReLU(0.2)
		)
		self.conv_block = nn.Sequential(
			nn.Conv2d(in_channel, base_channel, 3, 1, 1, bias=True),
			nn.LeakyReLU(0.2),
			nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
			nn.LeakyReLU(0.2),
			nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True)
		)

	def forward(self, x):
		x_ = self.conv_in(x)
		x_res = self.conv_block(x)
		return x_res+x_


class MaskNet(nn.Module):
	def __init__(self, in_channel, base_channel):
		super().__init__()
		self.block = ResBlock(in_channel, base_channel)
		self.out_conv = nn.Conv2d(base_channel, 1, 3, 1, 1, bias=True)
		self.sig = nn.Sigmoid()
		# self.backward_warp = BackwardWarp()

	def forward(self, im0, im1, i0tf, i1tf, t, f_t0, f_t1):
		h, w= im0.shape[2:]
		t = t.repeat(1, 1, h, w).float()
		# flow_in = F.interpolate(torch.cat((f_t0, f_t1), 1), scale_factor=0.25, mode='bilinear')
		data_in = torch.cat((im0, im1, i0tf, i1tf, t, f_t0, f_t1), 1)
		data_in_q = F.interpolate(data_in, scale_factor=0.25, mode='bilinear')
		# data_in_q = torch.cat((data_in_q, forward_feat, backward_feat), 1)
		masknet_out_q = self.out_conv(self.block(data_in_q))

		mask_f = F.interpolate(masknet_out_q, scale_factor=4, mode='bilinear')
		mask_full = mask_f[:, :1]
		# mask = self.sig(F.interpolate(mask+init_mask, scale_factor=4, mode='bilinear'))
		mask = self.sig(mask_full)
		return i0tf*(1-mask)+mask*i1tf