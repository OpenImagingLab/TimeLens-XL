import torch
import torch.nn as nn
from .encoder_module import EventsEncoder, ImageEncoder
from .flow_decoder import FlowDecoder, MaskNet
from tools.TimeTracker import Ttracker
from collections import OrderedDict


class FinalBidirectionAttenfusion(nn.Module):
	"""
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block,
    such as a ConvLSTM or a ConvGRU.
    Symmetric, skip connections on every encoding layer.

    num_block: the number of blocks in each simpleconvlayer.
    """

	def __init__(self, **kwargs):
		super().__init__()
		# self.interp_ratio = 16
		self.base_channel = kwargs['base_channel']
		self.pos_e = kwargs['pos_e']
		self.neg_e = kwargs['neg_e']
		self.echannel = kwargs['echannel']
		self.img_chn = kwargs['img_chn']
		self.num_decoder = kwargs['num_decoder']
		## event
		self.events_encoder = EventsEncoder(self.echannel, self.img_chn, self.base_channel,
		                                    num_decoder=self.num_decoder)
		self.im_encoder = ImageEncoder(self.img_chn // 2, self.base_channel)
		self.build_decoders()
		self.masknet = MaskNet(5 + self.img_chn * 2, self.base_channel * 4)
		self.cache_dict_init()
		self.tracker = Ttracker(False, 5)
		self.init_mask_dict = {}

	def build_decoders(self):
		self.decoder_0 = FlowDecoder(self.base_channel * 12, self.base_channel * 4, self.base_channel * 4)
		self.decoder_1 = FlowDecoder(self.base_channel * 12, self.base_channel * 4, self.base_channel * 4)

	def cache_dict_init(self):
		self.cache_dict = {
			'Flow_forward': [],
			'Flow_backward': [],
			'forward_ims': [],
			'backward_ims': [],
			'mask': []
		}

	# def interpolate(self, ):

	def forward(self, x, event, interp_ratio, end_time=None):
		"""
        :param x: b 2 c h w -> b, 2c, h, w
        :param event: b, t, num_bins, h, w -> b*t num_bins(2) h w
        :return: b, t, out_chn, h, w

        One direction propt version
        TODO:  use_reversed_voxel!!!
        """
		im0 = x[:, :self.img_chn // 2]
		im1 = x[:, self.img_chn // 2:]
		# reshape
		self.tracker.time_init()
		self.tracker.time_init_count()
		e_out, e_out_large = self.events_encoder(im0, im1, event, interp_ratio)
		im0_feat, im1_feat, im0_feat_full, im1_feat_full = self.im_encoder(x)
		self.tracker.track("Encoder")
		n, t, c, h, w = e_out.shape
		# im_expanded = torch.stack(self.im_conv(torch.stack((y0.unsqueeze(1), all_ecurs, y1.unsqueeze(1)), 0)).split(n, 0), 1)

		forward_ims = []
		backward_ims = []
		forward_cur_flow = torch.zeros((n, 2, h, w)).to(e_out.device)
		backward_cur_flow = torch.zeros((n, 2, h, w)).to(e_out.device)
		forward_cur_feat = im0_feat
		backward_cur_feat = im1_feat
		resout = []
		fuseout = []
		forward_Delta_flow_list, backward_Delta_flow_list = [], []
		forward_econv, backward_econv = im0_feat, im1_feat
		forward_flow_list = []
		backward_flow_list = []
		self.tracker.track("Prepare for decoder")
		end_time = range(t - 1) if end_time is None else end_time
		for t_ in range(max(end_time) + 1):
			forward_cur_feat, forward_cur_flow, forward_cur_flow_large, forward_econv = self.decoder_0(e_out[:, t_],
			                                                                                           im0_feat,
			                                                                                           forward_cur_flow,
			                                                                                           forward_cur_feat,
			                                                                                           forward_econv, )
			backward_cur_feat, backward_cur_flow, backward_cur_flow_large, backward_econv = self.decoder_1(
				e_out[:, t - t_ - 1],
				im1_feat,
				backward_cur_flow,
				backward_cur_feat,
				backward_econv)
			forward_ims.append(self.decoder_0.backwarp(im0, forward_cur_flow_large))
			backward_ims.insert(0, self.decoder_0.backwarp(im1, backward_cur_flow_large))
			forward_Delta_flow_list.append(forward_econv)
			backward_Delta_flow_list.insert(0, backward_econv)
			forward_flow_list.append(forward_cur_flow_large)
			backward_flow_list.insert(0, backward_cur_flow_large)
			self.tracker.track(f"decoder {t_}")

		_, _, lh, lw = forward_ims[0].shape
		mask_key = f"{lh}x{lw}"
		if mask_key not in self.init_mask_dict:
			self.init_mask_dict.update({
				mask_key: torch.ones((n, 1, lh, lw)).to(forward_ims[0].device)
			})
		tmask = self.init_mask_dict[mask_key]
		if end_time == range(t - 1):
			for t_ in range(t - 1):
				res, m, fo = self.masknet(im0,
				                          im1,
				                          forward_ims[t_],
				                          backward_ims[t_],
				                          (interp_ratio - t_ - 1) / interp_ratio * tmask,
				                          forward_flow_list[t_],
				                          backward_flow_list[t_],
				                          forward_Delta_flow_list[t_],
				                          backward_Delta_flow_list[t_],
				                          im0_feat_full,
				                          im1_feat_full
				                          )
				resout.append(res)
				self.tracker.track(f"mask {t_}")
				fuseout.append(fo)
		else:
			for t_ in end_time:
				res, m, fo = self.masknet(im0,
				                          im1,
				                          forward_ims[t_],
				                          backward_ims[t_ - end_time[0]],
				                          (interp_ratio - t_ - 1) / interp_ratio * tmask,
				                          forward_flow_list[t_],
				                          backward_flow_list[t_ - end_time[0]],
				                          forward_Delta_flow_list[t_],
				                          backward_Delta_flow_list[t_ - end_time[0]],
				                          im0_feat_full,
				                          im1_feat_full
				                          )
				resout.append(res)
				self.tracker.track(f"mask {t_}")
				fuseout.append(fo)
		self.tracker.count_plus()
		return torch.stack(resout, 1), torch.stack(forward_ims, 1), torch.stack(backward_ims, 1), \
			torch.stack(forward_Delta_flow_list, 1), torch.stack(backward_Delta_flow_list, 1), torch.stack(fuseout,
			                                                                                               1)  # b,t,c,h,w
