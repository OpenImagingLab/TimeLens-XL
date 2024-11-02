import torch
import sys
import os
sys.path.append(os.getcwd())
from thop import profile
from models.EVDI.EVDI import EVDI_Net
from params.GOPROv2eIMX636.params_trainGOPROVFI import trainGOPROVFI
from easydict import EasyDict as ED

args = ED()
args.model_name = 'EVDI_Net'
args.extension = ''
args.clear_previous = None
args.model_pretrained = None
args.calc_flops = True

params = trainGOPROVFI(args)
records = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{args.model_name}_flops_and_macs.txt'), 'a+')
params.training_config.crop_size = 256

for icount in range(2, 17, 2):
    params.training_config.interp_ratio = icount
    datashape_h, datashape_w = params.training_config.crop_size, params.training_config.crop_size

    net = EVDI_Net(params).cuda()
    left_frame = torch.randn(1, 1, datashape_h, datashape_w).float().cuda()
    right_frame = torch.randn(1, 1, datashape_h, datashape_w).float().cuda()
    previous_events = torch.randn(1, 32, datashape_h, datashape_w).float().cuda()
    post_events = torch.randn(1, 32, datashape_h, datashape_w).float().cuda()
    right_B_coef = torch.arange(1, params.training_config.interp_ratio).float() / params.training_config.interp_ratio
    right_B_coef = right_B_coef.view(-1, 1, 1, 1).cuda()
    left_B_coef = 1-right_B_coef

    macs, model_params = 0, 0
    for jcount in range(1, icount):
        outprofile = profile(net, inputs=(left_frame, right_frame, previous_events, post_events,
                                          left_B_coef[jcount-1:jcount],
                                          right_B_coef[jcount-1:jcount]))
        macs += outprofile[0]
        model_params += outprofile[1]
    content = f'[MODEL NAME] {args.model_name} '\
              f'[INPUT INFO] {datashape_h}x{datashape_w}x{params.training_config.interp_ratio} '\
              f'[MACs]       {macs/1e9:.3f} GMACs [AVERAGE MACs]: {macs/1e9/(params.training_config.interp_ratio-1):.3f} GMACs'\
              f'[PARAMs]     {model_params/1e9:.3f} G'
    print('-'*20)
    print(content)
    records.write(content+'\n')

