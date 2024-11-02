import torch
import numpy as np
import os
from tools.registery import DATASET_REGISTRY
from dataset.RC_4816.mixloader import MixLoader
import random
from numba import jit
from natsort import natsorted as sorted
from torch.nn import functional as F
from .dataset_dict import dataset_dict, test_key
from copy import deepcopy


@jit(nopython=True)
def trilinear_alloc_values(voxel, d_x, d_y, d_t, d_p, h, w, tstep, tstart):
    d_x_low, d_y_low = int(d_x), int(d_y)
    d_t_cur = (d_t-tstart)*tstep
    d_t_low = int(d_t_cur)

    x_weight = d_x - d_x_low
    y_weight = d_y - d_y_low
    t_weight = d_t_cur - d_t_low
    # pv = d_p * 2 - 1
    pv = d_p
    if d_y_low < h and d_x_low < w:
        voxel[d_t_low, d_y_low, d_x_low] += (1 - x_weight) * (1 - y_weight) * pv * (1 - t_weight)
        voxel[d_t_low+1, d_y_low, d_x_low] += (1 - x_weight) * (1 - y_weight) * pv * t_weight
    if d_y_low + 1 < h and d_x_low < w:
        voxel[d_t_low, d_y_low + 1, d_x_low] += (1 - x_weight) * y_weight * pv * (1 - t_weight)
        voxel[d_t_low+1, d_y_low + 1, d_x_low] += (1 - x_weight) * y_weight * pv * t_weight
    if d_x_low + 1 < w and d_y_low < h:
        voxel[d_t_low, d_y_low, d_x_low + 1] += (1 - y_weight) * x_weight * pv * (1 - t_weight)
        voxel[d_t_low+1, d_y_low, d_x_low + 1] += (1 - y_weight) * x_weight * pv * t_weight
    if d_y_low + 1 < h and d_x_low + 1 < w:
        voxel[d_t_low, d_y_low + 1, d_x_low + 1] += x_weight * y_weight * pv * (1 - t_weight)
        voxel[d_t_low+1, d_y_low + 1, d_x_low + 1] += x_weight * y_weight * pv * t_weight
    return


@jit(nopython=True)
def sample_events_to_grid(voxel_channels, h, w, x, y, t, p, hs, ws, tleft):
    ori_left_voxel = np.zeros((voxel_channels, h, w), dtype=np.float32)
    right_voxel = np.zeros((voxel_channels, h, w), dtype=np.float32)
    # if len(t) == 0:
    #     return voxel
    t_start = t[0]
    t_end = t[-1]
    # t_step = (t_end - t_start + 1) / voxel_channels
    # Compute left step
    tstep_left = float(voxel_channels-1)/float(tleft-t_start+1)
    tstep_right = float(voxel_channels-1) / float(t_end-tleft+1)
    for d in range(len(x)):
        d_x, d_y, d_t, d_p = x[d], y[d], t[d], p[d]
        if d_t < tleft:
            trilinear_alloc_values(ori_left_voxel, d_x, d_y, d_t, d_p, h, w, tstep_left, t_start)
        else:
            trilinear_alloc_values(right_voxel, d_x, d_y, d_t, d_p, h, w, tstep_right, tleft)
    left_voxel = -ori_left_voxel[::-1]
    return left_voxel, ori_left_voxel, right_voxel


@DATASET_REGISTRY.register()
class loader_RC_timelens_smallmix(MixLoader):
    def __init__(self, para, training=True):
        self.dataset_dict = deepcopy(dataset_dict)
        self.test_key = deepcopy(test_key)
        self.training_flag = training
        self.key = 'training_config' if self.training_flag else 'validation_config'
        self.crop_size = para[self.key]['crop_size']
        self.data_paths = para[self.key]['data_paths']
        self.path_dict = {}
        self.split_training_and_evaluation()
        print(self.path_dict.keys(), self.training_flag, self.dataset_dict.keys())
        self.num_bins = para.model_config.num_bins
        super().__init__(para, training)
        self.norm_voxel = True
        self.random_t = para[self.key]['random_t'] if training else False
        # self.sub_div = self.para.model_config.define_model.echannel//self.real_interp

        
    def events_reader(self, events_path, h, w, hs, ws, sample_t, interp_ratio):
        evs_data = [np.load(ep) for ep in events_path]
        ex, ey, ep, et = [], [], [], []
        for ed in evs_data:
            ex.extend(ed['x'])
            ey.extend(ed['y'])
            ep.extend(ed['p'])
            et.extend(ed['t'])
        ex, ey, ep, et = np.float32(ex), np.float32(ey), np.float32(ep), np.float32(et)
        if len(et) == 0:
            return torch.zeros((self.num_bins, h, w)).float(), torch.zeros((self.num_bins, h, w)).float(), torch.zeros((self.num_bins, h, w)).float()
        tsart = et[0]
        tend = et[-1]
        tstep = np.linspace(tsart, tend, interp_ratio+1)[1:-1]
        left_events, right_events, ori_left_events = [], [], []
        for st in sample_t:
            cur_timestamp = tstep[st-1]
            left_voxel, ori_left_voxel, right_voxel = sample_events_to_grid(self.num_bins, h, w, ex, ey, et, ep, hs, ws, cur_timestamp)

            left_events.append(left_voxel)
            ori_left_events.append(ori_left_voxel)
            right_events.append(right_voxel)
        # return torch.tensor(np.concatenate(evs_voxels, 0))
        return torch.from_numpy(np.stack(left_events, 0)), torch.from_numpy(np.stack(ori_left_events, 0)), torch.from_numpy(np.stack(right_events, 0))

    def data_loading(self, paths, sample_t, interp_ratio):
        folder_name, rgb_name, rgb_sample, evs_sample = paths
        im0 = self.imreader(rgb_sample[0])
        im1 = self.imreader(rgb_sample[-1])
        h, w = im0.shape[1:]
        events = self.events_reader(evs_sample, h, w, 0, 0, sample_t, interp_ratio)
        gts = [self.imreader(rgb_sample[st]) for st in sample_t]
        return folder_name, rgb_name, im0, im1, events, gts

    def __getitem__(self, item):
        item_content, interp_ratio = self.get_key_with_weights(item)
        if self.random_t:
            sample_t = [random.choice(range(1, interp_ratio))]
        else:
            sample_t = list(range(1, interp_ratio))
        folder_name, rgb_name, im0, im1, events, gts = self.data_loading(item_content, sample_t, interp_ratio)
        left_events, ori_left_events, right_events = events
        h, w = im0.shape[1:]
        if self.crop_size:
            hs, ws = random.randint(20, h - self.crop_size), random.randint(20, w - self.crop_size)
            im0, im1 = im0[:, hs:hs + self.crop_size, ws:ws + self.crop_size], im1[:, hs:hs + self.crop_size,
                                                                                       ws:ws + self.crop_size]
            left_events = left_events[..., hs:hs+self.crop_size, ws:ws+self.crop_size]
            right_events = right_events[..., hs:hs + self.crop_size, ws:ws + self.crop_size]
            ori_left_events = ori_left_events[..., hs:hs+self.crop_size, ws:ws+self.crop_size]
            gts = [gt[:, hs:hs + self.crop_size, ws:ws + self.crop_size] for gt in gts]
        else:
            # hn, wn = h//32*32, w//32*32
            hn, wn = (h//32-1)*32, (w//32-1)*32
            hleft = (h-hn)//2
            wleft = (w-wn)//2
            im0, im1 = im0[..., hleft:hleft+hn, wleft:wleft+wn], im1[..., hleft:hleft+hn, wleft:wleft+wn]
            left_events = left_events[..., hleft:hleft+hn, wleft:wleft+wn]
            ori_left_events = ori_left_events[..., hleft:hleft+hn, wleft:wleft+wn]
            right_events = right_events[..., hleft:hleft+hn, wleft:wleft+wn]
            gts = [gt[..., hleft:hleft+hn, wleft:wleft+wn] for gt in gts]
        gts = torch.cat(gts, 0)
        right_weight = [float(st) / interp_ratio for st in sample_t]
        rgb_name = [os.path.splitext(r)[0] for r in rgb_name]
        data_back = {
            'folder': os.path.split(folder_name)[-1],
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