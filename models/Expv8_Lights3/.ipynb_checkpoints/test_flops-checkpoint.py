import torch
import sys
import os
sys.path.append(os.getcwd())
from thop import profile
from models.Expv8_Lights3.runExpv8_Lights3 import Expv8_Lights3
from params.GOPRO_release.params_trainOurs_mix import trainGOPRO_Ours
from easydict import EasyDict as ED
import time

args = ED()
args.model_name = 'Expv8_Lights3'
args.extension = ''
args.clear_previous = None
args.model_pretrained = None
args.calc_flops = True

params = trainGOPRO_Ours(args)
records = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{args.model_name}_flops_and_macs.txt'), 'a+')
# params.training_config.crop_size = 64
params.save_flow = False
params.save_images = False

params.training_config.interp_ratio = 16
params.real_interp = 16
# params.real_interp = 4

params.debug = False
datashape_h, datashape_w = 608, 768
# datashape_h, datashape_w = params.training_config.crop_size, params.training_config.crop_size

net = Expv8_Lights3(params).cuda()
left_frame = torch.randn(3, 3, datashape_h, datashape_w).float().cuda()
right_frame = torch.randn(3, 3, datashape_h, datashape_w).float().cuda()
events = torch.randn(3, 128, datashape_h, datashape_w).float().cuda()

macs, model_params = 0, 0

outprofile = profile(net, inputs=(left_frame, right_frame, events, params.training_config.interp_ratio))
macs += outprofile[0]
model_params += outprofile[1]
content = f'[MODEL NAME] {args.model_name} '\
          f'[INPUT INFO] {datashape_h}x{datashape_w}x{params.training_config.interp_ratio} '\
          f'[MACs]       {macs/1e9:.3f} GMACs [AVERAGE MACs]: {macs/1e9/(params.real_interp-1):.3f} GMACs'\
          f'[PARAMs]     {model_params/1e9:.3f} G'
print('-'*20)
print(content)

with torch.no_grad():
    res = net(left_frame, right_frame, events, params.training_config.interp_ratio)
    t = time.time()
    for i in range(10):
        res = net(left_frame, right_frame, events, params.training_config.interp_ratio)
    td = time.time()-t
    print(td/10/(params.real_interp-1)/left_frame.shape[0])
print(left_frame.shape[0], td)
records.write(content+'\n')

