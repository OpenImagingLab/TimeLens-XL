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
CBMNet.train.rgb = r'E:\Research\EVS\Dataset\CBMNet\train' if hostname == 'local' else '/mnt/workspace/mayongrui/dataset/CBMNet/train'
CBMNet.train.evs = r'E:\Research\EVS\Dataset\Events_v2eIMX636v230809\train_x8interpolated' if hostname == 'local' else '/mnt/workspace/mayongrui/dataset/CBMNet/train_eventsvoxel'

CBMNet.test = ED()
CBMNet.test.rgb = r'E:\Research\EVS\Dataset\CBMNet\train' if hostname == 'local' else '/mnt/workspace/mayongrui/dataset/CBMNet/test'
CBMNet.test.evs = r'E:\Research\EVS\Dataset\Events_v2eIMX636v230809\test_x8interpolated' if hostname == 'local' else '/mnt/workspace/mayongrui/dataset/CBMNet/test_eventsvoxel'

CBMNet.train.evs_raw = r'E:\Research\EVS\Dataset\CBMNet\train' if hostname == 'local' else '/mnt/workspace/mayongrui/dataset/CBMNet/train'
CBMNet.test.evs_raw = r'E:\Research\EVS\Dataset\CBMNet\train' if hostname == 'local' else '/mnt/workspace/mayongrui/dataset/CBMNet/test'
