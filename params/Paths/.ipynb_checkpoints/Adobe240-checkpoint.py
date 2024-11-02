import os
from os.path import join, split, splitext
from tools import parse_path
import socket
from easydict import EasyDict as ED
import datetime


mkdir = lambda x:os.makedirs(x, exist_ok=True)


hostname = 'server' if 'PC' not in socket.gethostname() else 'local'

Adobe = ED()
Adobe.train = ED()
# BSERGB.train.rgb = '/mnt/workspace/mayongrui/dataset/Adobe240/3_TRAINING/'
# BSERGB.train.evs = '/mnt/workspace/mayongrui/dataset/bs_ergb/3_TRAINING/'


Adobe.test = ED()
Adobe.test.rgb = r'E:\Research\EVS\Dataset\bs_ergb\1_TEST' if hostname == 'local' else '/mnt/workspace/mayongrui/dataset/Adobe240FPS/RGB_dense'
Adobe.test.evs = r'E:\Research\EVS\Dataset\bs_ergb\1_TEST' if hostname == 'local' else '/mnt/workspace/mayongrui/dataset/Adobe240FPS/EVS_x8interpolatedct25v006/EVS'
