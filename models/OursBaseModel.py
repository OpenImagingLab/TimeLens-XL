from models.BaseModel import  BaseModel
import numpy as np
import torch
from tools.registery import MODEL_REGISTRY
import os
from torch.nn import functional as F


'''
Copy from Expv8_Lights2/runExpv8_Lights2.py
Make it a base class to eliminate duplicate code adjustments to ablations
'''

# @MODEL_REGISTRY.register()
class OursBase(BaseModel):
    def __init__(self, params):
        super().__init__(params)
        self.grad_cache = {}
        self.real_interp = None if 'real_interp' not in params.keys() else params.real_interp


    def rgb2y(self, x):
        x = (x[:, ::3] * 0.299
             + x[:, 1::3] * 0.587
             + x[:, 2::3] * 0.114)
        return x

    def save_training_samples(self, res, gt, events, data_in, epoch, step):
        from os import makedirs
        import os
        save_folder = os.path.join(self.train_im_path, str(epoch), str(step))
        makedirs(os.path.join(self.train_im_path, str(epoch)), exist_ok=True)
        makedirs(save_folder, exist_ok=True)
        file_names = data_in['rgb_name']
        N, B = len(file_names), len(file_names[0])
        for n in range(N):
            for b in range(B):
                if n == 0:
                    self.toim(data_in['im0'][b]).save(os.path.join(save_folder, f"b{b}n{n}_im0_id{file_names[n][b]}.jpg"))
                elif n == N-1:
                    self.toim(data_in['im1'][b]).save(os.path.join(save_folder, f"b{b}n{n}_im1_id{file_names[n][b]}.jpg"))
                else:
                    self.toim((res[b, n-1]).clamp(0, 1)).save(os.path.join(save_folder, f"b{b}n{n}_res_id{file_names[n][b]}.jpg"))
                    self.toim(gt[b, n-1]).save(os.path.join(save_folder, f"b{b}n{n}_gt_id{file_names[n][b]}.jpg"))
        return

    def cache_grad(self):
        for name, p in self.net.named_parameters():
            self.grad_cache.update({name:[p.grad.max(), p.grad.min()]})

    def resize_5d(self, data, scalar):
        n, t, c, h, w = data.shape
        data = F.interpolate(data.view(n*t, c, h, w), scale_factor=scalar, mode='bilinear')
        return data.view(n, t, c, int(h*scalar), int(w*scalar))

    def resize_4d(self, data, scalar):
        return F.interpolate(data, scale_factor=scalar, mode='bilinear')

    def epe(self, data, gt):
        return torch.mean(torch.sum((data-gt)**2, 1).sqrt())


    def net_training(self, data_in, optim, epoch, step):
        self.train()
        optim.zero_grad()
        left_frame, right_frame, events = data_in['im0'].cuda(), \
            data_in['im1'].cuda(), data_in['events'].cuda()
        interp_ratio = data_in['interp_ratio'][0].item()

        gts = data_in['gts'].cuda()
        # gts = self.rgb2y(gts)
        gts = gts.unsqueeze(2)
        scalar = interp_ratio-1 if self.real_interp is None else self.real_interp-1
        n, t, _, h, w = gts.shape
        # gts = torch.cat(gts.split(1, dim=1), dim=0)
        gts = gts.reshape(n, scalar, -1, h, w)
        c = gts.shape[2]
        res = self.forward(left_frame, right_frame, events, interp_ratio)
        recon = res[0]
        loss = self.update_training_metrics(recon, gts, epoch, step, optim.param_groups[0]['lr'], events, fuseout=res[-1])

        if torch.isnan(loss):
            with open('REFID_nan_log.txt', 'a+') as f:
                f.write('NAN loss happen!\n')
                f.write(f'Input Data Stats: , {left_frame.max()}, {left_frame.min()}, {right_frame.max()}, {right_frame.min()}, {events.max()}, {events.min()}')
            np.savez('REFID_nan_log', left_frame=left_frame.detach().cpu().numpy(),
                     right_frame=right_frame.detach().cpu().numpy(),
                     events=events.detach().cpu().numpy(),
                     ori_events=data_in['ori_events'].numpy())
            # torch.save(self.net.model_state(), f'{}')
            exit()
        loss.backward()

        # Causing Nan if remove gradient clip
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.01)

        optim.step()

        if epoch % self.train_im_save == 0 and step % self.train_print_freq == 0:
            self.save_training_samples(recon, gts, events, data_in, epoch, step)
        return

    def net_validation(self, data_in, epoch):
        self.eval()
        with torch.no_grad():
            left_frame, right_frame, events = data_in['im0'].cuda(), \
                data_in['im1'].cuda(), data_in['events'].cuda()
            interp_ratio = data_in['interp_ratio'].item()
            gts = data_in['gts'].cuda()

            # gts = self.rgb2y(gts)
            gts = gts.unsqueeze(2)
            scalar = interp_ratio - 1 if self.real_interp is None else self.real_interp - 1
            n, _, _, h, w = gts.shape
            gts = gts.reshape(n, scalar, -1, h, w)
            # gts = torch.cat(gts.split(1, dim=1), dim=0)

            # for n in range(gts.shape[0]):
            res = self.forward(left_frame,
                                 right_frame,
                                 events, interp_ratio)
            recon = res[0]
            flow = res[-2:]
            if self.debug:
                self.update_validation_metrics(recon, gts, epoch, data_in, cache_dict=self.net.cache_dict, flow=flow)
            else:
                self.update_validation_metrics(recon, gts, epoch, data_in, flow=flow)
        return

    def forward(self, left_frame, right_frame, events, interp_ratio):
        real_interp = interp_ratio if self.real_interp is None else self.real_interp
        jump_ratio = interp_ratio // real_interp
        end_tlist = range(jump_ratio-1, interp_ratio-1, jump_ratio)
        res = self.net(torch.cat((left_frame, right_frame), 1), events, interp_ratio, end_tlist)
        return res