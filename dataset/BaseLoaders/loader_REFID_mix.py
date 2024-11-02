import torch
from torch.nn.functional import interpolate
from torchvision.transforms import ToTensor, ToPILImage
import numpy as np
import glob
from natsort import natsorted as sorted
from torch.utils.data import Dataset
import os
from PIL import Image
import random
from tools.registery import DATASET_REGISTRY
from .mixbaseloader import MixBaseLoader


@DATASET_REGISTRY.register()
class loader_REFID_mix(MixBaseLoader):
    def __init__(self, para, training=True):
        super().__init__(para, training)
        self.norm_voxel = True

    def voxel_norm(self, voxel):
        """
        Norm the voxel

        :param voxel: The unnormed voxel grid
        :return voxel: The normed voxel grid
        """
        nonzero_ev = (voxel != 0)
        num_nonzeros = nonzero_ev.sum()
        # print('DEBUG: num_nonzeros:{}'.format(num_nonzeros))
        if num_nonzeros > 0:
            # compute mean and stddev of the **nonzero** elements of the event tensor
            # we do not use PyTorch's default mean() and std() functions since it's faster
            # to compute it by hand than applying those funcs to a masked array
            mean = voxel.sum() / num_nonzeros
            stddev = max(torch.sqrt((voxel ** 2).sum() / num_nonzeros - mean ** 2), 1e-4)
            mask = nonzero_ev.float()
            voxel = mask * (voxel - mean) / max(stddev, 1e-4)
        return voxel

    def events_temporal_acc(self, events_in, interp_ratio):
        events_out = torch.zeros((interp_ratio, events_in.shape[1], events_in.shape[2])).float()
        for i in range(interp_ratio):
            events_acc = torch.sum(events_in[i*self.rgb_sampling_ratio:(i+1)*self.rgb_sampling_ratio], dim=0)
            if self.norm_voxel:
                events_acc = self.voxel_norm(events_acc)
            events_out[i] = events_acc.clone()
        events_data_out = []
        for i in range(interp_ratio-1):
            events_data_out.append(events_out[i:i+2])
        return torch.stack(events_data_out, 0)

    def __getitem__(self, item):
        interp_ratio = self.weighted_random_selection()
        # interp_ratio = random.choice(self.interp_ratio)
        interp_ratio_key = str(interp_ratio)
        maxlen = len(self.total_file_indexing[interp_ratio_key]) - 1
        item_content = self.total_file_indexing[interp_ratio_key][min(item, maxlen)]
        sample_t = list(range(1, interp_ratio))
        folder_name, rgb_name, im0, im1, events, gts = self.data_loading(item_content, sample_t)

        h, w = im0.shape[1:]
        if self.crop_size:
            # hs, ws = 400, 800
            hs, ws = random.randint(0, h - self.crop_size), random.randint(0, w - self.crop_size)
            im0, im1, events = im0[:, hs:hs + self.crop_size, ws:ws + self.crop_size], im1[:, hs:hs + self.crop_size,
                                                                                       ws:ws + self.crop_size], events[
                                                                                                                :,
                                                                                                                hs:hs + self.crop_size,
                                                                                                                ws:ws + self.crop_size]
            gts = [gt[:, hs:hs + self.crop_size, ws:ws + self.crop_size] for gt in gts]
        else:
            h = h//32*32
            w = w//32*32
            im0, im1, events, gts = im0[..., :h, :w], im1[..., :h, :w], events[..., :h, :w], [gt[..., :h, :w] for gt in gts]
        gts = torch.cat(gts, 0)
        left_weight = [1 - float(st) / interp_ratio for st in sample_t]
        rgb_name = [os.path.splitext(r)[0] for r in rgb_name]
        ori_events = events.clone()
        events = self.events_temporal_acc(events, interp_ratio)

        data_back = {
            'folder': folder_name,
            'rgb_name': [rgb_name[0]] + [rgb_name[st] for st in sample_t] + [rgb_name[-1]],
            'im0': im0,
            'im1': im1,
            'gts': gts,
            'events': events,
            't_list': sample_t,
            'left_weight': left_weight,
            'ori_events':ori_events,
            'interp_ratio':interp_ratio
        }
        return data_back
