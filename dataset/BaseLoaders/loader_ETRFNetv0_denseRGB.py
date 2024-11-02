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
class loader_ERFNetv0_denseRGB(BaseLoader):
    def __init__(self, para, training=True):
        super().__init__(para, training)

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