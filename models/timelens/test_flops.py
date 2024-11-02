import torch
import sys
import os
sys.path.append(os.getcwd())
from thop import profile
from models.timelens.runtimelens import TimeLens
from params.GOPRO_release.params_trainTimeLens_attention import GOPRO_release_TimeLens_attention
from easydict import EasyDict as ED
import time


GOPRO_release_TimeLens_attention.training_stage = 'tuning'
args = ED()
args.model_name = 'TimeLens'
args.extension = ''
args.clear_previous = None
args.model_pretrained = None
args.calc_flops = True

params = GOPRO_release_TimeLens_attention(args)
records = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{args.model_name}_flops_and_macs.txt'), 'a+')
params.training_config.crop_size = 64

params.save_images = False
params.save_flow = False
params.training_config.interp_ratio = 2

params.debug = False
datashape_h, datashape_w = 608, 768
# datashape_h, datashape_w = params.training_config.crop_size, params.training_config.crop_size

net = TimeLens(params).cuda()
left_frame = torch.randn(5, 3, datashape_h, datashape_w).float().cuda()
right_frame = torch.randn(5, 3, datashape_h, datashape_w).float().cuda()
events = torch.randn(5, 128, datashape_h, datashape_w).float().cuda()

data_example = {
    'before':{
        'rgb_image_tensor':left_frame,
        'reversed_voxel_grid':torch.randn(5, 5, datashape_h, datashape_w).float().cuda(),
        'voxel_grid':torch.randn(5, 5, datashape_h, datashape_w).float().cuda()
    },
    'middle':{
        'weight':torch.randn(5, 1, 1, 1).float().cuda()
    },
    'after':{
        'rgb_image_tensor':right_frame,
        'voxel_grid':torch.randn(5, 5, datashape_h, datashape_w).float().cuda()
    }
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


with torch.no_grad():
    res = net(data_example)
    st = time.time()
    for i in range(10):
        net(data_example)
    runtime = time.time()-st
    content += f"[Time] {runtime/10*1e3/left_frame.shape[0]} ms"
print(content)

records.write(content+'\n')

