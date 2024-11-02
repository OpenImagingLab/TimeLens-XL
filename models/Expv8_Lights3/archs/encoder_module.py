import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

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
		return x_res + x_


class EventsEncoder(nn.Module):
	def __init__(self, e_channel, img_chn, base_channel, num_decoder=None):
		super().__init__()
		self.step = 8 if num_decoder is None else num_decoder
		# self.cur_step = e_channel // interp_ratio
		# self.pre_conv = ResBlock(e_channel+img_chn, e_channel)
		self.pre_conv = nn.Sequential(nn.Conv2d(e_channel+img_chn, e_channel, 3, 1, 1, bias=True), nn.LeakyReLU(0.2))
		self.ct=  nn.Parameter(torch.ones((1, e_channel, 1, 1))*0.2, requires_grad=True)
		self.down0 = nn.Sequential(
			nn.Conv2d(num_decoder, 2*base_channel, 3, 2, 1, bias=True),
			nn.LeakyReLU(0.2)
		)
		self.down1 = nn.Sequential(
			nn.Conv2d(2*base_channel, 4*base_channel, 3, 2, 1, bias=True),
			nn.LeakyReLU(0.2)
		)
		self.e_channel = e_channel
		self.padding_dict = {}

	def get_padding(self, b, c, h, w, device):
		k = f"{b}_{c}_{h}_{w}"
		if k not in self.padding_dict:
			self.padding_dict.update({
				k:torch.zeros((b, c, h, w), requires_grad=False).to(device).float()
			})
		return self.padding_dict[k]

	def forward(self, im0, im1, events, interp_ratio):
		# events_split_list = events.split(self.step, 1)
		# n, c, h, w = events_split_list[0].shape
		# events_stack = torch.cat(events_split_list, 0)
		# pre_conv = self.pre_conv(events_stack)
		cur_step = self.e_channel // interp_ratio
		pre_conv = self.pre_conv(torch.cat((im0, events, im1), 1))*self.ct
		if self.step == cur_step:
			events_split_list = pre_conv.split(self.step, 1)
			n, c, h, w = events_split_list[0].shape
		else:
			events_split_list_ = pre_conv.split(cur_step, 1)
			# n, c, h, w = events_split_list[0].shape
			# Need to squeeze the events
			if cur_step > self.step:
				sstep = cur_step // self.step
				evoxel = torch.stack(events_split_list_, 1)
				n, t, c, h, w = evoxel.shape
				evoxel = torch.sum(evoxel.view(n, t, sstep, self.step, h, w), 2)
				events_split_list = [evoxel[:, tc] for tc in range(t)]
			else:
				n, _, h, w = events_split_list_[0].shape
				paddings = self.get_padding(n, self.step-cur_step, h, w, im0.device)
				# paddings = torch.zeros((n, self.step-cur_step, h, w)).to(im0.device)
				events_split_list = [torch.cat((et, paddings), 1) for et in events_split_list_]
		events_stack = torch.cat(events_split_list, 0)
		# events_stack_sum = torch.sum(events_stack, 1, keepdim=True)
		e_half = self.down0(events_stack)
		# conv0 = self.conv0(e_half)
		e_quater = self.down1(e_half)
		# e_quater_conv = self.conv1(e_quater)
		_, _, ch, cw = e_quater.shape
		# e_out = e_quater_conv.view(n, t, -1, ch, cw)
		e_out = torch.stack(e_quater.split(n, 0), 1)
		return e_out, events_stack


class ImageEncoder(nn.Module):
	def __init__(self, img_ch, base_channel):
		super().__init__()
		self.img_ch = img_ch
		self.preblock = ResBlock(img_ch, base_channel)
		self.down0 = nn.Conv2d(base_channel, 2*base_channel, 3, 2, 1, bias=True)
		# self.conv0 = ResBlock(2*base_channel, 2*base_channel)
		self.down1 = nn.Conv2d(2*base_channel, 4*base_channel, 3, 2, 1, bias=True)
		# self.conv1 = ResBlock(4*base_channel, 4*base_channel)

	def forward(self, img):
		n, _, h, w = img.shape
		img = torch.cat(img.split(self.img_ch, 1), 0)
		preconv = self.preblock(img)
		im_half = self.down0(preconv)
		# conv0 = self.conv0(im_half)
		im_quater = self.down1(im_half)
		# im_quater_conv = self.conv1(im_quater)
		return im_quater[:n], im_quater[n:]