import os
from os.path import join, split, splitext
from tools import parse_path
import socket
from easydict import EasyDict as ED
import datetime


mkdir = lambda x:os.makedirs(x, exist_ok=True)


hostname = 'server' if 'PC' not in socket.gethostname() else 'local'

HSERGB = ED()

HSERGB.test = ED()
HSERGB.test.rgb = r'E:\Research\EVS\Dataset\hsergb' if hostname == 'local' else '/mnt/workspace/mayongrui/dataset/hsergb/'
HSERGB.test.evs = r'E:\Research\EVS\Dataset\hsergb' if hostname == 'local' else '/mnt/workspace/mayongrui/dataset/hsergb/'
