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
from .baseloader import BaseLoader


@DATASET_REGISTRY.register()
class loaderREFID_CBMNet(BaseLoader):
    def __init__(self, para, training=True):
        super().__init__(para, training)
        self.norm_voxel = True

        
    def ereader(self, events_path):
        evs_data = [self.totensor(np.load(ep, allow_pickle=True)['data'][0]) for ep in events_path]
        evs_data = torch.cat(evs_data, 0).float()
        return evs_data

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

    def events_temporal_acc(self, events_in):
        events_out = torch.zeros((self.interp_ratio, events_in.shape[1], events_in.shape[2])).float()
        for i in range(self.interp_ratio):
            events_acc = torch.sum(events_in[i*self.rgb_sampling_ratio:(i+1)*self.rgb_sampling_ratio], dim=0)
            if self.norm_voxel:
                events_acc = self.voxel_norm(events_acc)
            events_out[i] = events_acc.clone()
        events_data_out = []
        for i in range(self.interp_ratio-1):
            events_data_out.append(events_out[i:i+2])
        return torch.stack(events_data_out, 0)

    def __getitem__(self, item):
        item_content = self.samples_list[item]
        if self.random_t:
            sample_t = random.sample(range(1, self.interp_ratio // 2), self.sample_group // 2)
            sample_t.append(self.interp_ratio // 2)
            sample_t.extend(random.sample(range(self.interp_ratio // 2 + 1, self.interp_ratio), self.sample_group // 2))
        else:
            sample_t = list(range(1, self.interp_ratio))
        folder_name, rgb_name, im0, im1, events, gts = self.data_loading(item_content, sample_t)
        h, w = im0.shape[1:]
        if self.crop_size:
            hs, ws = random.randint(0, h - self.crop_size), random.randint(0, w - self.crop_size)
            im0, im1, events = im0[:, hs:hs + self.crop_size, ws:ws + self.crop_size], im1[:, hs:hs + self.crop_size,
                                                                                       ws:ws + self.crop_size], events[
                                                                                                                :,
                                                                                                                hs:hs + self.crop_size,
                                                                                                                ws:ws + self.crop_size]
            gts = [gt[:, hs:hs + self.crop_size, ws:ws + self.crop_size] for gt in gts]
        else:
            hn, wn = h//32*32, w//32*32
            im0, im1, events = im0[:, :hn, :wn], im1[:, :hn, :wn], events[:, :hn, :wn]
            gts = [gt[:, :hn, :wn] for gt in gts]
        gts = torch.cat(gts, 0)
        left_weight = [1 - float(st) / self.interp_ratio for st in sample_t]
        rgb_name = [os.path.splitext(r)[0] for r in rgb_name]
        ori_events = events.clone()
        events = self.events_temporal_acc(events)

        data_back = {
            'folder': folder_name,
            'rgb_name': [rgb_name[0]] + [rgb_name[st] for st in sample_t] + [rgb_name[-1]],
            'im0': im0,
            'im1': im1,
            'gts': gts,
            'events': events,
            't_list': sample_t,
            'left_weight': left_weight,
            'ori_events':ori_events
        }
        return data_back
