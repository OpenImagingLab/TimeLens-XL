import os
from os.path import join, split, splitext
from tools import parse_path
import socket
from easydict import EasyDict as ED
import datetime


mkdir = lambda x:os.makedirs(x, exist_ok=True)


hostname = 'server' if 'PC' not in socket.gethostname() else 'local'

GOPRO = ED()
GOPRO.train = ED()
# GOPRO.train.dense_rgb = r'E:\Research\EVS\Dataset\RGB_dense\test_x8interpolated'
GOPRO.train.dense_rgb = r'E:\Research\EVS\Dataset\RGB_dense\train_x8interpolated' if hostname == 'local' else '/mnt/workspace/mayongrui/dataset/GOPRO/RGB_dense/train_x8interpolated/'
GOPRO.train.rgb = '/home/PJLAB/mayongrui/Documents/Events/Dataset/videos/GOPRO_Large_all/RGB/train' if hostname == 'local' else '/mnt/data/oss_beijing/mayongrui/dataset/GOPRO_Large_all/RGB/train'
# Events in IMX 636 params, ct=0.25, ct_variance = 0.06
GOPRO.train.evs = r'E:\Research\EVS\Dataset\Events_v2eIMX636v230809\train_x8interpolated' if hostname == 'local' else '/mnt/workspace/mayongrui/dataset/GOPRO/Events_v2eIMX636v230809/train_x8interpolated/'
# Events in REFID params, ct=0.2, ct_variance=0.03
GOPRO.train.evs_refid = r'E:\Research\EVS\Dataset\Events_local\testing_events_ct20v003\EVS' if hostname == 'local' else '/mnt/workspace/mayongrui/dataset/GOPRO/EVS/train_x8interpolatedct25v006/EVS'

GOPRO.test = ED()
GOPRO.test.dense_rgb = r'E:\Research\EVS\Dataset\RGB_dense\test_x8interpolated' if hostname == 'local' else '/mnt/workspace/mayongrui/dataset/GOPRO/RGB_dense/test_x8interpolated/'
GOPRO.test.rgb = '/home/PJLAB/mayongrui/Documents/Events/Dataset/videos/GOPRO_Large_all/RGB/test' if hostname == 'local' else '/mnt/data/oss_beijing/mayongrui/dataset/GOPRO_Large_all/RGB/test'
# Events in IMX 636 params, ct=0.25, ct_variance=0.06
GOPRO.test.evs = r'E:\Research\EVS\Dataset\Events_v2eIMX636v230809\test_x8interpolated' if hostname == 'local' else '/mnt/workspace/mayongrui/dataset/GOPRO/Events_v2eIMX636v230809/test_x8interpolated/'
# Events in REFID params, ct=0.2, ct_variance = 0.03
GOPRO.test.evs_refid = r'E:\Research\EVS\Dataset\Events_local\testing_events_ct20v003\EVS' if hostname == 'local' else '/mnt/workspace/mayongrui/dataset/GOPRO/EVS/test_x8interpolatedct25v006/EVS'


GOPRO.challenge = ED()
GOPRO.challenge.dense_rgb = '/mnt/workspace/mayongrui/dataset/Challenges/selected_RGB_interpolation' if hostname != 'local' else r'E:\Research\EVS\Dataset\Challenges'
GOPRO.challenge.evs = '/mnt/workspace/mayongrui/dataset/Challenges/test_x8interpolatedct20v003/EVS'
GOPRO.challenge_training = ED()
GOPRO.challenge_training.dense_rgb = '/mnt/workspace/mayongrui/dataset/Challenges/training/selected_RGB_interpolation'
GOPRO.challenge_training.evs = '/mnt/workspace/mayongrui/dataset/Challenges/training/test_x8interpolatedct20v003/EVS'
