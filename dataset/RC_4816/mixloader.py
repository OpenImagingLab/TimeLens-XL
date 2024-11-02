import torch
import numpy as np
import os
from tools.registery import DATASET_REGISTRY
from dataset.BaseLoaders.mixbaseloader import MixBaseLoader
import random
from numba import jit
from natsort import natsorted as sorted
from .dataset_dict import dataset_dict, test_key, EVSneg3
from copy import deepcopy
from torchvision.transforms import ToPILImage, ToTensor
from torch.utils.data import Dataset
from PIL import Image


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


# @DATASET_REGISTRY.register()
class MixLoader(Dataset):
    def __init__(self, para, training=True):
        super().__init__()
        self.para = para
        self.training_flag = training
        self.key = 'training_config' if self.training_flag else 'validation_config'
        self.color = 'RGB' if 'color' not in para[self.key].keys() else para[self.key]['color']
        self.dataset_dict = deepcopy(dataset_dict)
        self.test_key = deepcopy(test_key)
        self.training_flag = training
        self.rgb_sampling_ratio = para[self.key]['rgb_sampling_ratio']
        self.key = 'training_config' if self.training_flag else 'validation_config'
        self.crop_size = para[self.key]['crop_size']
        self.data_paths = para[self.key]['data_paths']
        self.path_dict = {}
        self.samples_dict = {}
        self.split_training_and_evaluation()
        self.toim = ToPILImage()
        self.totensor = ToTensor()
        self.interp_ratio_list = [4, 8, 16, 32] if training else [para[self.key]['interp_ratio']]
        self.interp_list_pob = [0.15, 0.25, 0.5, 0.1] if training else [1]
        for irl in self.interp_ratio_list:
            self.samples_dict.update({
                str(irl):[]
            })
        self.samples_indexing()
        print(f'---- Interp List: {self.interp_ratio_list}, prob: {self.interp_list_pob}')
        self.norm_voxel = True
        self.echannel = 128 if 'define_model' not in self.para.model_config.keys() else self.para.model_config.define_model.echannel


    def split_training_and_evaluation(self,):
        folders = self.data_paths.keys()
        for folder in folders:
            curpath, ecurpath = self.data_paths[folder]
            if folder in self.dataset_dict or folder in EVSneg3:
                files = sorted(curpath)
                startIdx = int(os.path.split(files[0])[-1].split("_")[0])
                # filepath = [os.path.join(curpath, file) for file in files]
                filepath = files
                eventpath = sorted(ecurpath)
                sub_idx = 0
                if folder in self.dataset_dict:
                    data_list = self.dataset_dict[folder]
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
                else:
                    k = folder
                    self.path_dict.update({
                        k:[filepath, eventpath]
                    })
                    sub_idx += 1

    def samples_indexing(self):
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
            for irl in self.interp_ratio_list:
                for i_ind in range(0, len(indexes) - irl, 1 if self.training_flag else irl):
                    rgb_sample = [rgb_path[sind] for sind in indexes[i_ind:i_ind + irl + 1]]
                    evs_sample = evs_path[indexes[i_ind]:indexes[i_ind + irl]]
                    rgb_name = [os.path.splitext(os.path.split(rs)[-1])[0] for rs in rgb_sample]
                    self.samples_dict[str(irl)].append([k, rgb_name, rgb_sample, evs_sample])

        return

    def __len__(self):
        return len(self.samples_dict[str(self.interp_ratio_list[0])])

    def imreader(self, impath):
        im = self.totensor(Image.open(impath))
        if self.color == 'gray':
            im = im[0] * 0.299 + im[1] * 0.587 + im[2] * 0.114
            im = im.unsqueeze(0)
        return im

    def events_reader(self, events_path, h, w, interp_ratio):
        evs_data = [np.load(ep) for ep in events_path]
        evs_voxels = []
        for ed in evs_data:
            evs_voxels.append(
                sample_events_to_grid(self.echannel//interp_ratio, h, w, np.float32(ed['x']), np.float32(ed['y']), np.float32(ed['t']),
                                      np.float32(ed['p'])))
        return torch.tensor(np.concatenate(evs_voxels, 0))

    def data_loading(self, paths, sample_t, interp_ratio):
        folder_name, rgb_name, rgb_sample, evs_sample = paths
        im0 = self.imreader(rgb_sample[0])
        im1 = self.imreader(rgb_sample[-1])
        h, w = im0.shape[1:]
        events = self.events_reader(evs_sample, h, w, interp_ratio)
        gts = [self.imreader(rgb_sample[st]) for st in sample_t]
        return folder_name, rgb_name, im0, im1, events, gts

    def get_key_with_weights(self, idx):
        scalar = np.random.choice(self.interp_ratio_list, p=self.interp_list_pob)
        data_sample = self.samples_dict[str(scalar)][min(idx, len(self.samples_dict[str(scalar)])-1)]
        return data_sample, scalar

    def __getitem__(self, item):
        item_content, interp_ratio = self.get_key_with_weights(item)

        sample_t = list(range(1, interp_ratio))
        folder_name, rgb_name, im0, im1, events, gts = self.data_loading(item_content, sample_t, interp_ratio)
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
        left_weight = [1 - float(st) / interp_ratio for st in sample_t]
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
            'interp_ratio': interp_ratio
        }
        return data_back
