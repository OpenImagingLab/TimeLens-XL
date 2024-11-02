import torch
import sys
import os
sys.path.append(os.getcwd())
from thop import profile
from models.Expv8_large.runExpv8_large import Expv8_large
from params.GOPRO_release.params_trainOurs_mix import trainGOPRO_Ours
from easydict import EasyDict as ED
import time

args = ED()
args.model_name = 'Expv8_large'
args.extension = ''
args.clear_previous = None
args.model_pretrained = None
args.calc_flops = True

params = trainGOPRO_Ours(args)
records = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{args.model_name}_flops_and_macs.txt'), 'a+')
# params.training_config.crop_size = 64


params.training_config.interp_ratio = 16
params.real_interp = 16
# params.real_interp = 4
params.save_flow = False
params.save_images = False

params.debug = False
datashape_h, datashape_w = 608, 768
# datashape_h, datashape_w = params.training_config.crop_size, params.training_config.crop_size

net = Expv8_large(params).cuda()
left_frame = torch.randn(1, 3, datashape_h, datashape_w).float().cuda()
right_frame = torch.randn(1, 3, datashape_h, datashape_w).float().cuda()
events = torch.randn(1, 128, datashape_h, datashape_w).float().cuda()

macs, model_params = 0, 0

outprofile = profile(net, inputs=(left_frame, right_frame, events, 16))
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
    print((time.time()-t)/10/(params.real_interp-1))

records.write(content+'\n')

