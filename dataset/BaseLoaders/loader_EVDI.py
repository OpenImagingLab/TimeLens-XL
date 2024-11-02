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
class loader_EVDI(BaseLoader):
    def __init__(self, para, training=True):
        super().__init__(para, training)

    def ereader(self, events_path):
        evs_data = [self.totensor(np.load(ep, allow_pickle=True)['data']) for ep in events_path]
        evs_data = torch.cat(evs_data, 0).short()
        return evs_data

    #TODO: rewrite events channel to make it adjustable
    def events_dense_to_sparse(self, events_in, ind_t, events_channel=16):
        previous_events_out = torch.zeros(
            (events_channel * 2, events_in.shape[1], events_in.shape[2])).short()
        post_events_out = torch.zeros(
            (events_channel * 2, events_in.shape[1], events_in.shape[2])).short()
        previous_event = events_in[:ind_t, ...]
        previous_event = previous_event.flip(0)
        previous_index = np.linspace(0, ind_t, events_channel + 1)[1:]
        itind = 0
        for i in range(ind_t):
            if i > previous_index[itind]:
                itind += 1
            previous_events_out[itind][previous_event[i] > 0] += 1
            previous_events_out[itind + events_channel][previous_event[i] < 0] += 1
        post_event = events_in[ind_t:, ...]
        post_index = np.linspace(0, post_event.shape[0], events_channel + 1)[1:]
        itind = 0
        for i in range(post_event.shape[0]):
            if i > post_index[itind]:
                itind += 1
            post_events_out[itind][post_event[i] > 0] += 1
            post_events_out[itind + events_channel][post_event[i] < 0] += 1
        return previous_events_out, post_events_out


    def __getitem__(self, item):
        item_content = self.samples_list[item]
        folder_name, rgb_name, rgb_sample, evs_sample = item_content
        if self.random_t:
            sample_t = random.sample(range(1, self.interp_ratio // 2), self.sample_group // 2)
            sample_t.append(self.interp_ratio // 2)
            sample_t.extend(random.sample(range(self.interp_ratio // 2 + 1, self.interp_ratio), self.sample_group // 2))
        else:
            sample_t = list(range(1, self.interp_ratio))
        im0 = self.imreader(rgb_sample[0]).repeat(len(sample_t), 1, 1)
        im1 = self.imreader(rgb_sample[-1]).repeat(len(sample_t), 1, 1)
        events_in = self.ereader(evs_sample)
        gts = [self.imreader(rgb_sample[st]) for st in sample_t]


        if self.crop_size:
            h, w = im0.shape[1:]
            hs, ws = random.randint(0, h - self.crop_size), random.randint(0, w - self.crop_size)
            im0, im1 = im0[:, hs:hs + self.crop_size, ws:ws + self.crop_size], im1[:, hs:hs + self.crop_size, ws:ws + self.crop_size]
            events_in = events_in[:, hs:hs + self.crop_size, ws:ws + self.crop_size]
            gts = [gt[:, hs:hs + self.crop_size, ws:ws + self.crop_size] for gt in gts]

        previous_events = []
        post_events = []

        for t in sample_t:
            pree, pose = self.events_dense_to_sparse(events_in, t*self.rgb_sampling_ratio, 16)
            previous_events.append(pree)
            post_events.append(pose)

        previous_events = torch.cat(previous_events, 0).float()
        post_events = torch.cat(post_events, 0).float()
        gts = torch.cat(gts, 0)
        left_weight = [1 - float(st) / self.interp_ratio for st in sample_t]
        rgb_name = [os.path.splitext(r)[0] for r in rgb_name]
        data_back = {
            'folder': folder_name,
            'rgb_name': [rgb_name[0]] + [rgb_name[st] for st in sample_t] + [rgb_name[-1]],
            'im0': im0,
            'im1': im1,
            'gts': gts,
            'previous_events': previous_events,
            'post_events':post_events,
            't_list': sample_t,
            'left_weight': left_weight
        }
        return data_back
