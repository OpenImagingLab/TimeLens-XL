from models.BaseModel import  BaseModel
from models.REFID.archs import define_network
from easydict import EasyDict as ED
import numpy as np
import time
import torch
from tools.registery import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class REFID(BaseModel):
    def __init__(self, params):
        super().__init__(params)
        self.net = define_network(params.model_config.define_model)
        self.grad_cache = {}

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


    def net_training(self, data_in, optim, epoch, step):
        self.train()
        optim.zero_grad()
        left_frame, right_frame, events = data_in['im0'].cuda(), \
            data_in['im1'].cuda(), data_in['events'].cuda()
        interp_ratio = data_in['interp_ratio'][0].item()

        gts = data_in['gts'].cuda().unsqueeze(2)
        scalar = interp_ratio-1
        n, t, c, h, w = gts.shape
        # gts = torch.cat(gts.split(1, dim=1), dim=0)
        gts = gts.reshape(n, scalar, -1, h, w)
        recon = self.forward(left_frame, right_frame, events)
        loss = self.update_training_metrics(recon, gts, epoch, step, optim.param_groups[0]['lr'])
        if torch.isnan(loss):
            with open('REFID_nan_log.txt', 'a+') as f:
                f.write('NAN loss happen!\n')
                f.write(f'Input Data Stats: , {left_frame.max()}, {left_frame.min()}, {right_frame.max()}, {right_frame.min()}, {events.max()}, {events.min()}')
            np.savez('REFID_nan_log', left_frame=left_frame.detach().cpu().numpy(),
                     right_frame=right_frame.detach().cpu().numpy(),
                     events=events.detach().cpu().numpy(),
                     ori_events=data_in['ori_events'].numpy())
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
            interp_ratio = data_in['interp_ratio'][0].item()
            gts = data_in['gts'].cuda().unsqueeze(2)
            scalar = interp_ratio - 1
            n, _, _, h, w = gts.shape
            gts = gts.reshape(n, scalar, -1, h, w)
            # gts = torch.cat(gts.split(1, dim=1), dim=0)

            # for n in range(gts.shape[0]):
            recon = self.forward(left_frame,
                                 right_frame,
                                 events)
            self.update_validation_metrics(recon, gts, epoch, data_in)
        return

    def forward(self, left_frame, right_frame, events):
        return self.net(torch.cat((left_frame, right_frame), 1), events)
