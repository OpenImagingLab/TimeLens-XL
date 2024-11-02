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



class MixBaseLoader(Dataset):
    def __init__(self, para, training=True):
        self.para = para
        self.training_flag = training
        self.key = 'training_config' if self.training_flag else 'validation_config'
        self.crop_size = para[self.key]['crop_size']
        self.data_paths = para[self.key]['data_paths']
        self.data_index_offset = para[self.key]['data_index_offset']
        self.rgb_sampling_ratio = para[self.key]['rgb_sampling_ratio']
        self.interp_ratio = [para[self.key]['interp_ratio']] if self.key == 'validation_config' else [2, 4, 8, 16, 32]
        # self.sample_group = self.interp_ratio - 1 if 'sample_group' not in para[self.key].keys() else para[self.key]['sample_group']
        self.random_t = para[self.key]['random_t']
        self.toim = ToPILImage()
        self.totensor = ToTensor()

        self.total_file_indexing = {}
        for ir in self.interp_ratio:
            self.total_file_indexing.update({
                str(ir):[]
            })
        self.samples_indexing()
        self.color = 'gray' if 'color' not in para[self.key].keys() else para[self.key]['color']
        self.events_channel = None if 'events_channel' not in para.keys() else para.events_channel
        # self.events_channel
        item = 0
        item_content = self.total_file_indexing[list(self.total_file_indexing.keys())[0]][0]
        folder_name, rgb_name, rgb_sample, evs_sample = item_content
        print("!"*50)
        print("Using Mix data loader", self.interp_ratio)
        print(f'Training: {training}, EVS len {len(evs_sample)}')

    def samples_indexing(self):
        self.samples_list = []
        for k in self.data_paths.keys():
            rgb_path, evs_path = self.data_paths[k]
            indexes = list(range(0, len(rgb_path),
                                 self.rgb_sampling_ratio))
            for interp_ratio in self.interp_ratio:
                sample_list = []
                for i_ind in range(0, len(indexes) - interp_ratio, 1 if self.training_flag else interp_ratio):
                    rgb_sample = [rgb_path[sind] for sind in indexes[i_ind:i_ind + interp_ratio + 1]]
                    evs_sample = evs_path[indexes[i_ind]:indexes[i_ind + interp_ratio]]
                    rgb_name = [os.path.splitext(os.path.split(rs)[-1])[0] for rs in rgb_sample]
                    sample_list.append([k, rgb_name, rgb_sample, evs_sample])
                self.total_file_indexing[str(interp_ratio)].extend(sample_list)
        # self.samples_list = [self.samples_list[0]]
        return

    def __len__(self):
        k0 = list(self.total_file_indexing.keys())[0]
        return len(self.total_file_indexing[k0])

    def imreader(self, impath):
        im = self.totensor(Image.open(impath))
        if self.color == 'gray':
            im = im[0] * 0.299 + im[1] * 0.587 + im[2] * 0.114
            im = im.unsqueeze(0)
        return im

    def ereader(self, events_path):
        # evs_data = [self.totensor(np.load(ep, allow_pickle=True)['data']) for ep in events_path]
        single_event_channel = self.events_channel / len(events_path)
        evs_data = []
        for epath in events_path:
            edata = np.load(epath, allow_pickle=True)['data']
            h, w = edata.shape[-2:]
            if len(edata.shape) == 2:
                edata = np.expand_dims(edata, 0)
            c = edata.shape[0]

            if c <single_event_channel:
                edata = np.concatenate((edata, np.zeros((int(single_event_channel)-c, h, w), dtype=np.float32)), 0)
            evs_data.append(torch.from_numpy(edata))

        evs_data = torch.cat(evs_data, 0).float()
        if c > single_event_channel:
            acc_chn = c // single_event_channel
            evs_data = evs_data.view(-1, int(acc_chn), h, w).sum(1)
        return evs_data

    def adaptive_wei(self, ts, span_leftB, span_rightB):
        ## calculate weights, i.e., \omega & 1-\omega in paper
        ## This func coming from original code
        right = 0
        left = 0
        if (span_leftB[1] < ts) and (ts < span_rightB[0]):
            right = (ts - span_leftB[0]) / (span_rightB[1] - span_leftB[0])
            left = 1 - right
        if ts >= span_rightB[0]:
            right = 1
        if ts <= span_leftB[1]:
            left = 1
        sum_value = right + left
        right = right / sum_value
        left = left / sum_value
        return left, right

    def data_loading(self, paths, sample_t):
        folder_name, rgb_name, rgb_sample, evs_sample = paths
        im0 = self.imreader(rgb_sample[0])
        im1 = self.imreader(rgb_sample[-1])
        events = self.ereader(evs_sample)
        gts = [self.imreader(rgb_sample[st]) for st in sample_t]
        return folder_name, rgb_name, im0, im1, events, gts

    def weighted_random_selection(self):
        if len(self.interp_ratio) == 1:
            return self.interp_ratio[0]
        rand_num = random.random()
        if rand_num > 1 or rand_num < 0:
            print("Random Number Illegal", rand_num)
            exit()
        if rand_num < 0.1:
            return self.interp_ratio[0]
        elif rand_num > 0.1 and rand_num <= 0.2:
            return  self.interp_ratio[1]
        elif rand_num > 0.2 and rand_num <= 0.4:
            return self.interp_ratio[2]
        elif rand_num > 0.4 and rand_num <= 0.7:
            return self.interp_ratio[3]
        else:
            return self.interp_ratio[4]

    def __getitem__(self, item):
        interp_ratio = self.weighted_random_selection()
        interp_ratio_key = str(interp_ratio)
        maxlen = len(self.total_file_indexing[interp_ratio_key])-1
        item_content = self.total_file_indexing[interp_ratio_key][min(item, maxlen)]

        sample_t = list(range(1, interp_ratio))
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
        left_weight = [1 - float(st) / interp_ratio for st in sample_t]
        rgb_name = [os.path.splitext(r)[0] for r in rgb_name]
        data_back = {
            'folder': folder_name,
            'rgb_name': [rgb_name[0]] + [rgb_name[st] for st in sample_t] + [rgb_name[-1]],
            'im0': im0,
            'im1': im1,
            'gts': gts,
            'events': events,
            't_list': sample_t,
            'left_weight': left_weight,
            'interp_ratio':interp_ratio
        }
        return data_back
