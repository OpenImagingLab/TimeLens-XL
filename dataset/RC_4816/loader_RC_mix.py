import torch
import numpy as np
import os
from tools.registery import DATASET_REGISTRY
from dataset.RC_4816.mixloader import MixLoader




@DATASET_REGISTRY.register()
class mix_loader_smallRC(MixLoader):
    def __init__(self, para, training=True):
        super().__init__(para, training)
