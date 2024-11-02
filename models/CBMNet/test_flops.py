import torch
import sys
import os
sys.path.append(os.getcwd())
from thop import profile
from models.CBMNet.runmycbmnet import myCBMNet
# from models.CBMNet.runcbmnet_large import CBMNet_large
from params.GOPRO_release.params_trainCBMNet_joint import GOPRO_release_CBMNet_joint
from easydict import EasyDict as ED
import time



args = ED()
args.model_name = 'myCBMNet'
args.extension = ''
args.clear_previous = None
args.model_pretrained = None
args.calc_flops = True

params = GOPRO_release_CBMNet_joint(args)
records = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{args.model_name}_flops_and_macs.txt'), 'a+')
params.training_config.crop_size = 64

params.save_flow = False
params.save_images = False
params.training_config.interp_ratio = 2

params.debug = False
datashape_h, datashape_w = 608, 768
# datashape_h, datashape_w = params.training_config.crop_size, params.training_config.crop_size

net = myCBMNet(params).cuda()
left_frame = torch.randn(2, 3, datashape_h, datashape_w).float().cuda()
right_frame = torch.randn(2, 3, datashape_h, datashape_w).float().cuda()
events = torch.randn(2, 16, datashape_h, datashape_w).float().cuda()

data_example = {
            "image_input0": left_frame,
            "image_input1": right_frame,
            'event_input_0t': events,
            'event_input_t1': events,
            'event_input_t0': events
    }


macs, model_params = 0, 0

outprofile = profile(net, inputs=(data_example,))
macs += outprofile[0]
model_params += outprofile[1]
content = f'[MODEL NAME] {args.model_name} '\
          f'[INPUT INFO] {datashape_h}x{datashape_w}x{params.training_config.interp_ratio} '\
          f'[MACs]       {macs/1e9:.3f} GMACs [AVERAGE MACs]: {macs/1e9/(params.training_config.interp_ratio-1):.3f} GMACs'\
          f'[PARAMs]     {model_params/1e9:.3f} G'
print('-'*20)
print(content)
# res = net(data_example)
net = net.eval()
content = ''
with torch.no_grad():
    net(data_example)
    st = time.time()
    for i in range(10):
        net(data_example)
        print(i, time.time()-st)
    runtime = time.time()-st
    content += f"[Time] {runtime/10*1e3/2} ms"
print(content)

records.write(content+'\n')

