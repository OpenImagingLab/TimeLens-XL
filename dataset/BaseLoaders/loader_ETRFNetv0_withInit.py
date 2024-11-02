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
import cv2



@DATASET_REGISTRY.register()
class loader_ERFNetv0_withInit(BaseLoader):
    def __init__(self, para, training=True):
        super().__init__(para, training)
        self.disflow = cv2.DISOpticalFlow()
        self.disflow.create(1)

    def calc_flow(self, im0, im1):
        im0 = np.uint8(im0.clamp(0, 1)[0, 0]*255)
        im1 = np.uint8(im1.clamp(0, 1)[0, 0]*255)
        flow = self.disflow.calc(im0, im1, None)
        return self.totensor(flow)

    def __getitem__(self, item):
        item_content = self.samples_list[item]
        folder_name, rgb_name, rgb_sample, evs_sample = item_content
        if self.random_t:
            sample_t = random.sample(range(1, self.interp_ratio // 2), self.sample_group // 2)
            sample_t.append(self.interp_ratio // 2)
            sample_t.extend(random.sample(range(self.interp_ratio // 2 + 1, self.interp_ratio), self.sample_group // 2))
        else:
            sample_t = list(range(1, self.interp_ratio))
        im0 = self.imreader(rgb_sample[0])
        im1 = self.imreader(rgb_sample[-1])
        events = self.ereader(evs_sample)
        gts = [self.imreader(rgb_sample[st]) for st in sample_t]
        if self.crop_size:
            h, w = im0.shape[1:]
            hs, ws = random.randint(0, h - self.crop_size), random.randint(0, w - self.crop_size)
            im0, im1, events = im0[:, hs:hs + self.crop_size, ws:ws + self.crop_size], im1[:, hs:hs + self.crop_size,
                                                                                       ws:ws + self.crop_size], events[
                                                                                                                :,
                                                                                                                hs:hs + self.crop_size,
                                                                                                                ws:ws + self.crop_size]
            gts = [gt[:, hs:hs + self.crop_size, ws:ws + self.crop_size] for gt in gts]
        gts = torch.cat(gts, 0)
        left_weight = [1 - float(st) / self.interp_ratio for st in sample_t]
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
            'flow':self.calc_flow(im0, im1)
        }
        return data_back
