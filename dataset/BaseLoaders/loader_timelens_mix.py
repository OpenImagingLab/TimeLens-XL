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
class loader_timelens_mix(MixBaseLoader):
    def __init__(self, para, training=True):
        super().__init__(para, training)
        self.num_bins = para.model_config.num_bins
        print("!"*50)
        print(f"Using Mix Interp", self.interp_ratio)

    def ereader(self, events_path):
        evs_data = [np.load(ep, allow_pickle=True)['data'] for ep in events_path]
        # evs_data = torch.cat(evs_data, 0).float()
        evs_data = np.stack(evs_data, 0).astype(np.float32)
        return evs_data

    def generate_right_events(self, right_events):
        t, h, w= right_events.shape
        events_out = np.zeros((self.num_bins, h, w), dtype=np.float32)
        if t <= self.num_bins:
            events_out[:t] = right_events
        else:
            tstep = (self.num_bins-1)/t
            for t_ind in range(t):
                tcur = t_ind * tstep
                lt = int(tcur)
                ht = lt+1
                events_out[lt] += right_events[t_ind] * (ht - tcur)
                events_out[ht] += right_events[t_ind] * (tcur - lt)
        return events_out

    def generate_left_events(self, left_events):
        ori_left_events = np.copy(left_events)
        # left_events = -left_events[::-1]
        t, h, w= ori_left_events.shape
        # events_out = np.zeros((self.num_bins, h, w), dtype=np.float32)
        ori_events_out = np.zeros((self.num_bins, h, w), dtype=np.float32)
        if t <= self.num_bins:
            ori_events_out[:t] = left_events
        else:
            tstep = (self.num_bins-1)/t
            # for t_ind in range(t):
            #     tcur = t_ind * tstep
            #     lt = int(tcur)
            #     ht = lt+1
            #     events_out[lt] += left_events[t_ind] * (ht - tcur)
            #     events_out[ht] += left_events[t_ind] * (tcur - lt)
            for t_ind in range(t):
                tcur = t_ind * tstep
                lt = int(tcur)
                ht = lt + 1
                ori_events_out[lt] += ori_left_events[t_ind] * (ht - tcur)
                ori_events_out[ht] += ori_left_events[t_ind] * (tcur - lt)

        return -ori_events_out[::-1], ori_events_out


    def events_dense2bins(self, events, sample_t):
        left_events = []
        ori_left_events = []
        right_events = []
        for st in sample_t:
            left_event, ori_left_event = self.generate_left_events(events[:st*self.rgb_sampling_ratio])
            left_events.append(left_event)
            ori_left_events.append(ori_left_event)
            right_events.append(self.generate_right_events(events[st*self.rgb_sampling_ratio:]))
        return torch.from_numpy(np.stack(left_events, 0)).float(), torch.from_numpy(np.stack(ori_left_events, 0)).float(), torch.from_numpy(np.stack(right_events, 0)).float()

    def data_loading(self, paths, sample_t):
        folder_name, rgb_name, rgb_sample, evs_sample = paths
        im0 = self.imreader(rgb_sample[0])
        im1 = self.imreader(rgb_sample[-1])
        events = self.ereader(evs_sample)
        events = self.events_dense2bins(events, sample_t)
        gts = [self.imreader(rgb_sample[st]) for st in sample_t]
        return folder_name, rgb_name, im0, im1, events, gts

    def __getitem__(self, item):
        interp_ratio = self.weighted_random_selection()
        # interp_ratio = random.choice(self.interp_ratio)
        interp_ratio_key = str(interp_ratio)
        maxlen = len(self.total_file_indexing[interp_ratio_key]) - 1
        item_content = self.total_file_indexing[interp_ratio_key][min(item, maxlen)]
        if self.random_t:
            sample_t = [random.choice(range(1, interp_ratio))]
        else:
            sample_t = list(range(1, interp_ratio))
        folder_name, rgb_name, im0, im1, events, gts = self.data_loading(item_content, sample_t)
        left_events, ori_left_events, right_events = events
        h, w = im0.shape[1:]
        if self.crop_size:
            hs, ws = random.randint(0, h - self.crop_size), random.randint(0, w - self.crop_size)
            im0, im1 = im0[:, hs:hs + self.crop_size, ws:ws + self.crop_size], im1[:, hs:hs + self.crop_size,
                                                                                       ws:ws + self.crop_size]
            left_events = left_events[..., hs:hs+self.crop_size, ws:ws+self.crop_size]
            right_events = right_events[..., hs:hs + self.crop_size, ws:ws + self.crop_size]
            ori_left_events = ori_left_events[..., hs:hs+self.crop_size, ws:ws+self.crop_size]
            gts = [gt[:, hs:hs + self.crop_size, ws:ws + self.crop_size] for gt in gts]
        else:
            hn, wn = h//32*32, w//32*32
            im0, im1 = im0[..., :hn, :wn], im1[..., :hn, :wn]
            left_events = left_events[..., :hn, :wn]
            ori_left_events = ori_left_events[..., :hn, :wn]
            right_events = right_events[..., :hn, :wn]
            gts = [gt[..., :hn, :wn] for gt in gts]
        gts = torch.cat(gts, 0)
        right_weight = [float(st) / interp_ratio for st in sample_t]
        rgb_name = [os.path.splitext(r)[0] for r in rgb_name]
        data_back = {
            'folder': folder_name,
            'rgb_name': [rgb_name[0]] + [rgb_name[st] for st in sample_t] + [rgb_name[-1]],
            'im0': im0,
            'im1': im1,
            'gts': gts,
            'left_events': left_events.squeeze(),
            'ori_left_events':ori_left_events.squeeze(),
            'right_events':right_events.squeeze(),
            't_list': sample_t,
            'right_weight': right_weight
        }
        return data_back