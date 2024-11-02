import os
from os.path import join, split, splitext
from tools import parse_path
import socket
from easydict import EasyDict as ED
import datetime


mkdir = lambda x:os.makedirs(x, exist_ok=True)


hostname = 'server' if 'PC' not in socket.gethostname() else 'local'

CBMNet = ED()
CBMNet.train = ED()
CBMNet.train.rgb = '/home/PJLAB/mayongrui/Documents/Events/Dataset/videos/GOPRO_Large_all/RGB/train' if hostname == 'local' else '/mnt/data/sail_3090/mayongrui/dataset/CBMNet/train'
CBMNet.train.evs = r'E:\Research\EVS\Dataset\Events_v2eIMX636v230809\train_x8interpolated' if hostname == 'local' else '/mnt/data/sail_3090/mayongrui/dataset/CBMNet/train_eventsvoxel'

CBMNet.test = ED()
CBMNet.test.rgb = '/home/PJLAB/mayongrui/Documents/Events/Dataset/videos/GOPRO_Large_all/RGB/test' if hostname == 'local' else '/mnt/data/sail_3090/mayongrui/dataset/CBMNet/test'
CBMNet.test.evs = r'E:\Research\EVS\Dataset\Events_v2eIMX636v230809\test_x8interpolated' if hostname == 'local' else '/mnt/data/sail_3090/mayongrui/dataset/CBMNet/test_eventsvoxel'


