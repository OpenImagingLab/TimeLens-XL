import torch
from torch import nn
import numpy as np
from .warp_layer import warp
from torch.nn import functional as F


class ResBlock(nn.Module):
	def __init__(self, in_channel, base_channel):
		super().__init__()
		self.conv_in = nn.Sequential(
			nn.Conv2d(in_channel, base_channel, 3, 1, 1, bias=True),
		)
		self.conv_block = nn.Sequential(
			nn.LeakyReLU(0.2),
			nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
			nn.LeakyReLU(0.2),
			nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
			nn.LeakyReLU(0.2),
			nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
			nn.LeakyReLU(0.2),
		)

	def forward(self, x):
		x_ = self.conv_in(x)
		# x_ = self.conv_in(x)
		x_res = self.conv_block(x_)
		return x_res+x_
    
class ResBlockIF(nn.Module):
	def __init__(self, in_channel, base_channel):
		super().__init__()
		self.conv_in = nn.Sequential(
			nn.Conv2d(in_channel, base_channel, 3, 1, 1, bias=True),
		)
		self.conv_block = nn.Sequential(
			nn.LeakyReLU(0.2),
			nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
			nn.LeakyReLU(0.2),
			nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
			nn.LeakyReLU(0.2),
			nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
			nn.LeakyReLU(0.2),
		)
		self.conv_block1 = nn.Sequential(
			nn.LeakyReLU(0.2),
			nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
			nn.LeakyReLU(0.2),
			nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
			nn.LeakyReLU(0.2),
			nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
			nn.LeakyReLU(0.2),
		)

	def forward(self, x):
		x_ = self.conv_in(x)
		# x_ = self.conv_in(x)
		x_res = self.conv_block(x_)+x_
		x_res1 = self.conv_block1(x_res)+x_res
		return x_res1


class BackwardWarp(nn.Module):
	def __init__(self):
		super().__init__()
		self.grid_dict = {}

	def update_grid(self, key, N, H, W):
		gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))

		gridX = torch.tensor(gridX, requires_grad=False, ).cuda().unsqueeze(0).repeat(N, 1, 1).float()
		gridY = torch.tensor(gridY, requires_grad=False, ).cuda().unsqueeze(0).repeat(N, 1, 1).float()
		self.grid_dict.update({
			key:[gridX, gridY]
		})
		return

	def forward(self, img, flow):
		N, _, H, W = img.size()

		u = flow[:, 0, :, :]
		v = flow[:, 1, :, :]

		grid_key = f"{N}_{H}_{W}"
		if grid_key not in self.grid_dict:
			self.update_grid(grid_key, N, H, W)
		gridX, gridY = self.grid_dict[grid_key]

		x = gridX + u
		y = gridY + v
		# range -1 to 1
		x = 2 * (x / W - 0.5)
		y = 2 * (y / H - 0.5)
		# stacking X and Y
		grid = torch.stack((x, y), dim=3)
		# Sample pixels using bilinear interpolation.
		imgOut = torch.nn.functional.grid_sample(img, grid, align_corners=False)

		return imgOut

class FlowDecoder(nn.Module):
	def __init__(self, in_channel, base_channel, conv_base_channel):
		super().__init__()
		self.block_1 = ResBlock(in_channel, base_channel)
		# self.block_2 = ResBlock(base_channel*4+2, conv_base_channel)
		self.block_3 = ResBlock(base_channel*2+2, conv_base_channel)
		self.block_4 = ResBlock(base_channel*4+2, conv_base_channel)
		# self.conv_out0 = nn.Conv2d(conv_base_channel, 2, 3, 1, 1, bias=True)
		# self.conv_out1 = nn.Conv2d(conv_base_channel, 2, 3, 1, 1, bias=True)
		self.conv_out2 = nn.Conv2d(conv_base_channel, 2, 3, 1, 1, bias=True)
		self.conv_out3 = nn.Conv2d(conv_base_channel, 2, 3, 1, 1, bias=True)
		self.backwarp = BackwardWarp()

	def forward(self, events, im1, last_flow, last_warp_res, prev_econv):
		encoder_in = torch.cat((last_warp_res, events, prev_econv), 1)
		# Estimate Fti->ti+1
		block_1 = self.block_1(encoder_in)
		# conv_out0 = self.conv_out0(block_1)
		# warp_res0 = self.backwarp(last_warp_res, conv_out0)

		# block_2_incat = torch.cat((last_warp_res, events, warp_res0, conv_out0, block_1), 1)
		# block_2 = self.block_2(block_2_incat)
		# between_flow = self.conv_out1(block_2)+conv_out0
		# im_tiplus = self.backwarp(last_warp_res, between_flow)
		im_tiplus = block_1

		# Flow Fusion
		border_flow_in = torch.cat((last_flow, im1, im_tiplus), 1)
		block_3 = self.block_3(border_flow_in)
		flow0 = self.conv_out2(block_3)+last_flow
		warp_res = self.backwarp(im1, flow0)

		final_flow_in = torch.cat((im_tiplus, im1, flow0, warp_res, block_3), 1)
		block_4 = self.block_4(final_flow_in)
		flow = self.conv_out3(block_4)+flow0

		out_warp_res = self.backwarp(im1, flow)
		flow_large = F.interpolate(flow, scale_factor=4, mode='bilinear')*4
		return out_warp_res, flow, flow_large, im_tiplus

	def backwarp_(self, img, flow):
		_, _, H, W = img.size()

		u = flow[:, 0, :, :]
		v = flow[:, 1, :, :]

		gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))

		gridX = torch.tensor(gridX, requires_grad=False, ).cuda()
		gridY = torch.tensor(gridY, requires_grad=False, ).cuda()
		x = gridX.unsqueeze(0).expand_as(u).float() + u
		y = gridY.unsqueeze(0).expand_as(v).float() + v
		# range -1 to 1
		x = 2 * (x / W - 0.5)
		y = 2 * (y / H - 0.5)
		# stacking X and Y
		grid = torch.stack((x, y), dim=3)
		# Sample pixels using bilinear interpolation.
		imgOut = torch.nn.functional.grid_sample(img, grid)

		return imgOut
    
class IFBlock(nn.Module):
	def __init__(self, in_channel, base_channel):
		super().__init__()
		self.bwarp = BackwardWarp()
		self.resConv = ResBlockIF(in_channel, base_channel)
		self.out_conv = nn.Conv2d(base_channel, 5, 3, 1, 1, bias=True)
        
	def forward(self, i0, i1, i0tf, i1tf, ft0, ft1, m, t, scale, forward_feat=None, backward_feat=None):
		data_in = torch.cat((i0, i1, i0tf, i1tf), 1)
		if m is not None:
			data_in = torch.cat((data_in, m), 1)
		else:
			data_in = torch.cat((data_in, t), 1)
		data_in = torch.cat((data_in, ft0, ft1), 1)
		data_in_lowres = F.interpolate(data_in, scale_factor=scale, mode='bilinear')
		if forward_feat is not None:
			data_in_lowres = torch.cat((data_in_lowres, forward_feat, backward_feat), 1)
		after_conv_lowres = self.out_conv(self.resConv(data_in_lowres))
		after_conv_highres = F.interpolate(after_conv_lowres, scale_factor=1/scale, mode='bilinear')
		Ft0 = ft0+after_conv_highres[:, 1:3]
		Ft1 = ft1+after_conv_highres[:, 3:]
		Mask = after_conv_highres[:, :1] if m is None else after_conv_highres[:, :1]+m
		I0tf = self.bwarp(i0, Ft0)
		I1tf = self.bwarp(i1, Ft1)
		return Ft0, Ft1, I0tf, I1tf, Mask


class MaskNet(nn.Module):
	def __init__(self, in_channel, base_channel):
		super().__init__()
		# self.block = ResBlock(in_channel, base_channel)
		# self.out_conv = nn.Conv2d(base_channel, 5, 3, 1, 1, bias=True)
		self.sig = nn.Sigmoid()
		# self.backward_warp = BackwardWarp()
		self.quater_block = IFBlock(in_channel+2*base_channel, base_channel)
		self.half_block = IFBlock(in_channel, base_channel//2)
		self.channel_squeeze0 = nn.Conv2d(128, 32, 1, 1, 0, bias=True)
		self.channel_squeeze1 = nn.Conv2d(128, 32, 1, 1, 0, bias=True)
		self.refine_blocks = ResBlockIF(3+128, 64)
		self.conv_out = nn.Conv2d(64, 3, 1, 1, bias=True)

	def forward(self, im0, im1, i0tf, i1tf, t, f_t0, f_t1, forward_feat, backward_feat, im0_feat_full, im1_feat_full):
		qft0, qft1, qi0tf, qi1tf, qm = self.quater_block(im0, im1, i0tf, i1tf, f_t0, f_t1, None, t, 0.25, forward_feat, backward_feat)
		hft0, hft1, hi0tf, hi1tf, hm = self.half_block(im0, im1, qi0tf, qi1tf, qft0, qft1, qm, t, 0.5)
		mask = self.sig(hm)
		fuse_out = hi0tf*mask+(1-mask)*hi1tf

		forward_feat_ld = self.channel_squeeze0(forward_feat)
		backward_feat_ld = self.channel_squeeze1(backward_feat)
		forward_feat_hd, backward_feat_hd = F.interpolate(forward_feat_ld, scale_factor=4, mode='bilinear'), F.interpolate(backward_feat_ld, scale_factor=4, mode='bilinear')
		refine_in = torch.cat((forward_feat_hd, backward_feat_hd, fuse_out, im0_feat_full, im1_feat_full), 1)
		refine_out = self.conv_out(self.refine_blocks(refine_in))+fuse_out
		return refine_out, hft0, hft1