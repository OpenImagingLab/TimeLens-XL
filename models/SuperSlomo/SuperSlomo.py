from time import time
import click
import torch
import numpy as np
from models.SuperSlomo import model
from torchvision import transforms
from torch.functional import F
from torch import nn
import os



class SuperSlomoModel(nn.Module):
	def __init__(self, params):
		super().__init__()
		mean = [0.429, 0.431, 0.397]
		mea0 = [-m for m in mean]
		std = [1] * 3
		self.trans_forward = transforms.Compose([transforms.Normalize(mean=mean, std=std)])
		self.trans_backward = transforms.Compose([transforms.Normalize(mean=mea0, std=std)])

		self.flow = model.UNet(6, 4)
		self.interp = model.UNet(20, 5)
		self.back_warp = None
		self.interp_ratio = params.validation_config.interp_ratio
		self.load_models(os.path.join(os.getcwd(), 'pretrained_weights/SuperSloMo.ckpt'))

	def setup_back_warp(self, w, h, device):
		# global back_warp
		with torch.set_grad_enabled(False):
			self.back_warp = model.backWarp(w, h, device).to(device)

	def load_models(self, checkpoint):
		states = torch.load(checkpoint, map_location='cpu')
		self.interp.load_state_dict(states['state_dictAT'])
		self.flow.load_state_dict(states['state_dictFC'])

	def forward(self, frame0, frame1):
		i0 = self.trans_forward(frame0)
		i1 = self.trans_forward(frame1)
		h, w= i0.shape[2:]
		self.setup_back_warp(w, h, i0.device)
		ix = torch.cat([i0, i1], dim=1)

		flow_out = self.flow(ix)
		f01 = flow_out[:, :2, :, :]
		f10 = flow_out[:, 2:, :, :]

		frame_buffer = []
		for i in range(1, self.interp_ratio):
			t = i / self.interp_ratio
			temp = -t * (1 - t)
			co_eff = [temp, t * t, (1 - t) * (1 - t), temp]

			ft0 = co_eff[0] * f01 + co_eff[1] * f10
			ft1 = co_eff[2] * f01 + co_eff[3] * f10

			gi0ft0 = self.back_warp(i0, ft0)
			gi1ft1 = self.back_warp(i1, ft1)

			iy = torch.cat((i0, i1, f01, f10, ft1, ft0, gi1ft1, gi0ft0), dim=1)
			io = self.interp(iy)

			ft0f = io[:, :2, :, :] + ft0
			ft1f = io[:, 2:4, :, :] + ft1
			vt0 = F.sigmoid(io[:, 4:5, :, :])
			vt1 = 1 - vt0

			gi0ft0f = self.back_warp(i0, ft0f)
			gi1ft1f = self.back_warp(i1, ft1f)

			co_eff = [1 - t, t]

			ft_p = (co_eff[0] * vt0 * gi0ft0f + co_eff[1] * vt1 * gi1ft1f) / \
			       (co_eff[0] * vt0 + co_eff[1] * vt1)

			frame_buffer.append(self.trans_backward(ft_p))

		return frame_buffer
