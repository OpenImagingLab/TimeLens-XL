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
class loader_ERFNetv0(BaseLoader):
    def __init__(self, para, training=True):
        super().__init__(para, training)
