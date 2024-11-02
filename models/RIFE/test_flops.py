import torch
import sys
import os
sys.path.append(os.getcwd())
from thop import profile
from models.RIFE.runRIFE import RIFE
from params.Adobe.params_testRGBx16 import Adobe_release_RGB_testx16
from easydict import EasyDict as ED
import time



args = ED()
args.model_name = 'RIFE'
args.extension = ''
args.clear_previous = None
args.model_pretrained = None
args.calc_flops = True

params = Adobe_release_RGB_testx16(args)
records = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{args.model_name}_flops_and_macs.txt'), 'a+')
# params.training_config.crop_size = 64
args.training = False
params.save_images = False
params.save_flow = False
params.training_config.interp_ratio = 16

params.debug = False
datashape_h, datashape_w = 720//32*32, 1280//32*32
# datashape_h, datashape_w = params.training_config.crop_size, params.training_config.crop_size

net = RIFE(params).cuda()
left_frame = torch.randn(2, 3, datashape_h, datashape_w).float().cuda()
right_frame = torch.randn(2, 3, datashape_h, datashape_w).float().cuda()
events = torch.randn(2, 128, datashape_h, datashape_w).float().cuda()

macs, model_params = 0, 0
print("Interp ratio", )
with torch.no_grad():
    net.forward(left_frame, right_frame, events,)
print("Successful run")
# outprofile = profile(net.forward, inputs=(left_frame, right_frame, events,))
# macs += outprofile[0]
# model_params += outprofile[1]
# content = f'[MODEL NAME] {args.model_name} '\
#           f'[INPUT INFO] {datashape_h}x{datashape_w}x{params.training_config.interp_ratio} '\
#           f'[MACs]       {macs/1e9:.3f} GMACs [AVERAGE MACs]: {macs/1e9/(params.training_config.interp_ratio-1):.3f} GMACs'\
#           f'[PARAMs]     {model_params/1e9:.3f} G'
# print('-'*20)
# print(content)
# print(outprofile)
content = ''

with torch.no_grad():
    res = net(left_frame, right_frame, events)
    st = time.time()
    for i in range(10):
        net(left_frame, right_frame, events)
    runtime = time.time()-st
    content += f"[Time] {runtime/10*1e3/15} ms"
print(content)

records.write(content+'\n')

