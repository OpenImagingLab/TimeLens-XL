import os
from os.path import join, split, splitext
from tools import parse_path
import socket
from easydict import EasyDict as ED
import datetime


mkdir = lambda x:os.makedirs(x, exist_ok=True)


# hostname = 'server' if 'PC' not in socket.gethostname() else 'local'
hostname = 'yongrui'
BSERGB = ED()
BSERGB.train = ED()
# BSERGB.train.rgb = r'E:\Research\EVS\Dataset\bs_ergb\3_TRAINING' if hostname == 'local' else '/mnt/workspace/mayongrui/dataset/bs_ergb/3_TRAINING/'
# BSERGB.train.evs = r'E:\Research\EVS\Dataset\bs_ergb\3_TRAINING' if hostname == 'local' else '/mnt/workspace/mayongrui/dataset/bs_ergb/3_TRAINING/'
BSERGB.train.rgb = '/media/mayongrui/DataSet/bs_ergb/3_TRAINING'
BSERGB.train.evs = '/media/mayongrui/DataSet/bs_ergb/3_TRAINING'


BSERGB.test = ED()
# BSERGB.test.rgb = r'E:\Research\EVS\Dataset\bs_ergb\1_TEST' if hostname == 'local' else '/mnt/workspace/mayongrui/dataset/bs_ergb/1_TEST/'
# BSERGB.test.evs = r'E:\Research\EVS\Dataset\bs_ergb\1_TEST' if hostname == 'local' else '/mnt/workspace/mayongrui/dataset/bs_ergb/1_TEST/'
BSERGB.test.rgb = '/media/mayongrui/DataSet/bs_ergb/1_TEST'
BSERGB.test.evs = '/media/mayongrui/DataSet/bs_ergb/1_TEST'

