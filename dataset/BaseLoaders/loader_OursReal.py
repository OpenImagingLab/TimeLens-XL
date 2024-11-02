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
from numba import jit


@jit(nopython=True)
def sample_events_to_grid(voxel_channels, h, w, x, y, t, p):
    voxel = np.zeros((voxel_channels, h, w), dtype=np.float32)
    t_start = t[0]
    t_end = t[-1]
    t_step = (t_end - t_start + 1) / voxel_channels
    for d in range(len(x)):
        d_x, d_y, d_t, d_p = x[d], y[d], t[d], p[d]
        d_x_low, d_y_low = int(d_x), int(d_y)
        d_t = d_t - t_start

        x_weight = d_x - d_x_low
        y_weight = d_y - d_y_low
        ind = int(d_t // t_step)
        voxel[ind, d_y_low, d_x_low] += (1 - x_weight) * (1 - y_weight) * d_p
        if d_y_low + 1 < h:
            voxel[ind, d_y_low + 1, d_x_low] += (1 - x_weight) * y_weight * d_p
        if d_x_low + 1 < w:
            voxel[ind, d_y_low, d_x_low + 1] += (1 - y_weight) * x_weight * d_p
        if d_y_low + 1 < h and d_x_low + 1 < w:
            voxel[ind, d_y_low + 1, d_x_low + 1] += x_weight * y_weight * d_p
    return voxel


@DATASET_REGISTRY.register()
class loaderREFID_OursReal(BaseLoader):
    def __init__(self, para, training=True):
        super().__init__(para, training)
        self.norm_voxel = True
        self.sub_div = self.para.model_config.define_model.echannel//self.interp_ratio

    def samples_indexing(self):
        self.samples_list = []
        for k in self.data_paths.keys():
            rgb_path, evs_path = self.data_paths[k]
            indexes = list(range(0, len(rgb_path),
                                 self.rgb_sampling_ratio))
            for i_ind in range(0, len(indexes) - self.interp_ratio, 1 if self.training_flag else self.interp_ratio):
                # print(i_ind, self.interp_ratio, len(indexes), indexes[0], indexes[-1], len(rgb_path), len(evs_path))
                rgb_sample = [rgb_path[sind] for sind in indexes[i_ind:i_ind + self.interp_ratio + 1]]
                evs_sample = evs_path[indexes[i_ind]:indexes[i_ind + self.interp_ratio]]
                rgb_name = [os.path.splitext(os.path.split(rs)[-1])[0] for rs in rgb_sample]
                self.samples_list.append([k, rgb_name, rgb_sample, evs_sample])
        # self.samples_list = [self.samples_list[0]]
        return

    def events_reader(self, events_path, h, w):
        evs_data = [np.load(ep) for ep in events_path]
        evs_voxels = []
        for ed in evs_data:
            evs_voxels.append(sample_events_to_grid(self.sub_div, h, w, ed['x'], ed['y'], ed['t'], ed['p']))
        return torch.tensor(np.concatenate(evs_voxels, 0))

    def data_loading(self, paths, sample_t):
        folder_name, rgb_name, rgb_sample, evs_sample = paths
        im0 = self.imreader(rgb_sample[0])
        im1 = self.imreader(rgb_sample[-1])
        h, w = im0.shape[1:]
        events = self.events_reader(evs_sample, h, w)
        gts = [self.imreader(rgb_sample[st]) for st in sample_t]
        return folder_name, rgb_name, im0, im1, events, gts

