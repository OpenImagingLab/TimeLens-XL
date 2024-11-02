import argparse
import torch
import numpy as np
import params
from .registery import PARAM_REGISTRY
from datetime import datetime
import os

def keyword_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--param_name", type=str, default="trainGOPROVFI", help="model saving path")
    parser.add_argument("--model_name", type=str, default="E_TRFNetv0", help="model name")
    parser.add_argument("--seed", type=int, default=1226, help="random seed")
    parser.add_argument("--model_pretrained", type=str, default=None, help="model saving name")
    parser.add_argument("--init_step", type=int, default=None, help="initialize training steps")
    parser.add_argument("--skip_training", action="store_true", help="Whether or not enable training")
    parser.add_argument("--clear_previous", action="store_true", help="Delete previous results")
    parser.add_argument("--extension", type=str, default='', help="extension of save folder")
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--calc_flops", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save_flow", type=bool, default=False, help="save optical flow or not")
    parser.add_argument("--save_images", type=bool, default=True, help="save images for validation or not")

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    params = PARAM_REGISTRY.get(args.param_name)(args)
    params.args = args
    # params.use_cuda = args.use_cuda
    params.gpu_ids = range(torch.cuda.device_count())
    params.enable_training = not args.skip_training
    params.init_step = args.init_step
    params.local_rank = args.local_rank
    params.debug = args.debug
    params.save_flow = args.save_flow
    params.save_images = args.save_images

    print('Training and testing using parameters: ', args.param_name)
    print('Model: ', params.model_config.name)
    print(f"GPU ids: {params.gpu_ids}")
    return params
