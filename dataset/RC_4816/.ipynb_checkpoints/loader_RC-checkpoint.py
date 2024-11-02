import torch
import numpy as np
import os
from tools.registery import DATASET_REGISTRY
from dataset.BaseLoaders.baseloader import BaseLoader
import random
from numba import jit
from natsort import natsorted as sorted
from .dataset_dict import dataset_dict, test_key
from copy import deepcopy

@jit(nopython=True)
def sample_events_to_grid(voxel_channels, h, w, x, y, t, p):
    voxel = np.zeros((voxel_channels, h, w), dtype=np.float32)
    if len(t) == 0:
        return voxel
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
        # pv = d_p * 2 - 1
        pv = d_p
        if d_y_low < h and d_x_low < w:
            voxel[ind, d_y_low, d_x_low] += (1 - x_weight) * (1 - y_weight) * pv
        if d_y_low + 1 < h and d_x_low < w:
            voxel[ind, d_y_low + 1, d_x_low] += (1 - x_weight) * y_weight * pv
        if d_x_low + 1 < w and d_y_low < h:
            voxel[ind, d_y_low, d_x_low + 1] += (1 - y_weight) * x_weight * pv
        if d_y_low + 1 < h and d_x_low + 1 < w:
            voxel[ind, d_y_low + 1, d_x_low + 1] += x_weight * y_weight * pv
    return voxel


@DATASET_REGISTRY.register()
class loader_RC(BaseLoader):
    def __init__(self, para, training=True):
        self.real_interp = para.real_interp
        self.dataset_dict = deepcopy(dataset_dict)
        self.test_key = deepcopy(test_key)
        self.training_flag = training
        self.key = 'training_config' if self.training_flag else 'validation_config'
        self.crop_size = para[self.key]['crop_size']
        self.data_paths = para[self.key]['data_paths']
        self.path_dict = {}
        self.split_training_and_evaluation()
        super().__init__(para, training)
        self.norm_voxel = True
        self.echannel = 128 if 'define_model' not in self.para.model_config.keys() else self.para.model_config.define_model.echannel
        self.sub_div = self.echannel // self.real_interp


    def split_training_and_evaluation(self,):
        folders = self.data_paths.keys()
        for folder in folders:
            curpath, ecurpath = self.data_paths[folder]
            if folder in self.dataset_dict:
                files = sorted(curpath)
                data_list = self.dataset_dict[folder]
                startIdx = int(os.path.split(files[0])[-1].split("_")[0])
                # filepath = [os.path.join(curpath, file) for file in files]
                filepath = files
                eventpath = sorted(ecurpath)
                sub_idx = 0
                while len(data_list) > 0:
                    start = data_list.pop(0)
                    end = data_list.pop(0)
                    start = start - startIdx
                    end = end - startIdx
                    k = f"{folder}_{sub_idx}" if sub_idx > 0 else folder
                    self.path_dict.update({
                        k: [filepath[start:end], eventpath[start:end]]
                    })
                    sub_idx += 1

    def samples_indexing(self):
        self.samples_list = []
        for k in self.path_dict.keys():
            if self.training_flag:
                if k not in self.test_key:
                    rgb_path, evs_path = self.path_dict[k]
                else:
                    rgb_path, evs_path = [], []
            else:
                if k in self.test_key:
                    rgb_path, evs_path = self.path_dict[k]
                else:
                    rgb_path, evs_path = [], []
            indexes = list(range(0, len(rgb_path)-1,
                                 self.rgb_sampling_ratio))
            for i_ind in range(0, len(indexes) - self.interp_ratio, 1 if self.training_flag else self.interp_ratio):
                rgb_sample = [rgb_path[sind] for sind in indexes[i_ind:i_ind + self.interp_ratio + 1]]
                evs_sample = evs_path[indexes[i_ind]:indexes[i_ind + self.interp_ratio]]
                rgb_name = [os.path.splitext(os.path.split(rs)[-1])[0] for rs in rgb_sample]
                self.samples_list.append([k, rgb_name, rgb_sample, evs_sample])
        return

    def events_reader(self, events_path, h, w):
        evs_data = [np.load(ep) for ep in events_path]
        evs_voxels = []
        for ed in evs_data:
            evs_voxels.append(
                sample_events_to_grid(self.sub_div, h, w, np.float32(ed['x']), np.float32(ed['y']), np.float32(ed['t']),
                                      np.float32(ed['p'])))
        return torch.tensor(np.concatenate(evs_voxels, 0))

    def data_loading(self, paths, sample_t):
        folder_name, rgb_name, rgb_sample, evs_sample = paths
        im0 = self.imreader(rgb_sample[0])
        im1 = self.imreader(rgb_sample[-1])
        h, w = im0.shape[1:]
        events = self.events_reader(evs_sample, h, w)
        gts = [self.imreader(rgb_sample[st]) for st in sample_t]
        return folder_name, rgb_name, im0, im1, events, gts

    def __getitem__(self, item):
        item_content = self.samples_list[item]
        if self.random_t:
            sample_t = random.sample(range(1, self.real_interp // 2), self.sample_group // 2)
            sample_t.append(self.real_interp // 2)
            sample_t.extend(random.sample(range(self.real_interp // 2 + 1, self.real_interp), self.sample_group // 2))
        else:
            sample_t = list(range(1, self.real_interp))
        folder_name, rgb_name, im0, im1, events, gts = self.data_loading(item_content, sample_t)
        h, w = im0.shape[1:]
        if self.crop_size:
            hs, ws = random.randint(20, h - self.crop_size - 20), random.randint(20, w - self.crop_size - 20)
            im0, im1, events = im0[:, hs:hs + self.crop_size, ws:ws + self.crop_size], im1[:, hs:hs + self.crop_size,
                                                                                       ws:ws + self.crop_size], events[
                                                                                                                :,
                                                                                                                hs:hs + self.crop_size,
                                                                                                                ws:ws + self.crop_size]
            gts = [gt[:, hs:hs + self.crop_size, ws:ws + self.crop_size] for gt in gts]
        else:
            hn, wn = (h // 32 - 1) * 32, (w // 32 - 1) * 32
            hleft = (h - hn) // 2
            wleft = (w - wn) // 2
            im0, im1, events = im0[:, hleft:hleft + hn, wleft:wleft + wn], im1[:, hleft:hleft + hn,
                                                                           wleft:wleft + wn], events[:,
                                                                                              hleft:hleft + hn,
                                                                                              wleft:wleft + wn]
            gts = [gt[:, hleft:hleft + hn, wleft:wleft + wn] for gt in gts]
        gts = torch.cat(gts, 0)
        left_weight = [1 - float(st) / self.real_interp for st in sample_t]
        rgb_name = [os.path.splitext(r)[0] for r in rgb_name]
        data_back = {
            'folder': os.path.split(folder_name)[-1],
            'rgb_name': [rgb_name[0]] + [rgb_name[st] for st in sample_t] + [rgb_name[-1]],
            'im0': im0,
            'im1': im1,
            'gts': gts,
            'events': events,
            't_list': sample_t,
            'left_weight': left_weight,
            'interp_ratio': self.interp_ratio
        }
        return data_back
