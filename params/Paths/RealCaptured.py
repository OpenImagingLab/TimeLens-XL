import os
from os.path import join, split, splitext
from tools import parse_path
import socket
from easydict import EasyDict as ED
import datetime


mkdir = lambda x:os.makedirs(x, exist_ok=True)

hostname = 'yongrui'
RC = ED()
RC.train = ED()
RC.train.rgb = '/media/mayongrui/DataSet/HQ-EVFI'
RC.train.evs = '/media/mayongrui/DataSet/HQ-EVFI'

RC.test = ED()
RC.test.rgb = '/media/mayongrui/DataSet/HQ-EVFI'
RC.test.evs = '/media/mayongrui/DataSet/HQ-EVFI'
