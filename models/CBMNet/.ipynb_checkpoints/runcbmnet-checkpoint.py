import sys
import os
sys.path.append(os.path.join(os.getcwd(), "models/CBMNet/"))
from models.BaseModel import  BaseModel
import numpy as np
import torch
from tools.registery import MODEL_REGISTRY
from models.CBMNet.cbmnet_models.final_models.ours import EventInterpNet
from torch.nn import functional as F
import time


@MODEL_REGISTRY.register()
class CBMNet(BaseModel):
    def __init__(self, params):
        super().__init__(params)
        self.training_stage = params.training_stage
        self.net = EventInterpNet().cuda()
        self.net.training_stage = params.training_stage
        self.net.debug = self.debug
        # self.params_training = self.net.params_training
        self.grad_cache = {}
        try:
            from models.raft import raft
            self.raft = raft().cuda()
        except:
            pass


    def rgb2y(self, x):
        x = (x[:, ::3] * 0.299
             + x[:, 1::3] * 0.587
             + x[:, 2::3] * 0.114)
        return x
    
    
    def update_training_metrics_cbmnet_flow(self, res, res_flow, gt, epoch, step, lr, *args):
        loss = 0
        print_content = f"MODEL {self.params.model_config.name}\tCur EPOCH/STEP/LR: [{epoch}/{step}/{lr:.6f}]\t"
        self.metrics_record["training_time"].append(time.time())
        if step % self.train_print_freq == 0:
            print_content += f'TIme: {self.metrics_record["training_time"][1]-self.metrics_record["training_time"][0]:.4f}\t' \
                if step == 0 else f'TIme: {self.metrics_record["training_time"][step]-self.metrics_record["training_time"][step-self.train_print_freq]:.4f}\t'
            if step == 0:
                self.metrics_record['training_time'].pop()

        for k in self.training_metrics.keys():
            for mi in range(2):
                func, as_loss = self.training_metrics[k]
                if k == 'ECharbonier':
                    events =  args[0]
                    loss_item = func.forward(res[mi], gt, events)
                elif k == 'SmoothLoss':
                    loss_item = func.forward(res_flow[mi], gt)
                elif k == 'dists' or k=='lpips':
                    if len(res[mi].shape) == 5:
                        rn, rt, rc, rh, rw = res[mi].shape
                        loss_item = func.forward(res[mi].view(-1, rc, rh, rw), gt.view(-1, rc, rh, rw))
                    else:
                        loss_item = func.forward(res[mi], gt)
                else:
                    loss_item = func.forward(res[mi], gt)
                if as_loss:
                    loss += loss_item
            self.metrics_record[f"train_{k}"].append(loss_item.item())
            # print_content += f'{k}: {loss_item.item():.6f}\t'
            if step % self.train_print_freq == 0:
                print_content += f'{k}: {self.metrics_record[f"train_{k}"][-1]:.4f}\t' \
                    if step == 0 else f'{k}: {np.mean(self.metrics_record[f"train_{k}"][step-self.train_print_freq:step]):.4f}\t'
        if step % self.train_print_freq == 0:
            print(print_content)
            with open(os.path.join(os.path.join(self.train_im_path, str(epoch)+'.txt')), 'a+') as f:
                f.write(print_content+'\n')
        return loss

    def save_training_samples(self, res, gt, data_in, epoch, step):
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
                    self.toim((res[b]).clamp(0, 1)).save(os.path.join(save_folder, f"b{b}n{n}_res_id{file_names[n][b]}.jpg"))
                    self.toim(gt[b]).save(os.path.join(save_folder, f"b{b}n{n}_gt_id{file_names[n][b]}.jpg"))
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

    def pack_data(self, data_in):
        return {
            "image_input0": data_in['im0'].cuda(),
            "image_input1":data_in['im1'].cuda(),
            'event_input_0t': data_in['ori_left_events'].cuda(),
            'event_input_t1': data_in['right_events'].cuda(),
            'event_input_t0': data_in['left_events'].cuda()
        }

    def pack_valdata(self, data_in, ind):
        return {
            "image_input0": data_in['im0'].cuda(),
            "image_input1":data_in['im1'].cuda(),
            'event_input_0t': data_in['ori_left_events'][:, ind].cuda(),
            'event_input_t1': data_in['right_events'][:, ind].cuda(),
            'event_input_t0': data_in['left_events'][:, ind].cuda()
        }

    def net_training(self, data_in, optim, epoch, step):
        self.train()
        optim.zero_grad()
        data_sample = self.pack_data(data_in)

        gts = data_in['gts'].cuda()
        res = self.forward(data_sample)
        recon = res[0]
        res_flow = res[1] if self.training_stage == 'flow' else None
        if self.training_stage == 'flow':
            loss = self.update_training_metrics_cbmnet_flow(recon, res_flow, gts, epoch, step, optim.param_groups[0]['lr'])
        else:
            loss = self.update_training_metrics(recon, res_flow, gts, epoch, step, optim.param_groups[0]['lr'])

        loss.backward()

        # Causing Nan if remove gradient clip
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.01)

        optim.step()

        if epoch % self.train_im_save == 0 and step % self.train_print_freq == 0:
            self.save_training_samples((recon[0]+recon[1])/2, gts, data_in, epoch, step)
        return

    def net_validation(self, data_in, epoch):
        self.eval()
        with torch.no_grad():
            # left_frame, right_frame, events = data_in['im0'].cuda(), \
            #     data_in['im1'].cuda(), data_in['events'].cuda()
            # data_sample = self.pack_data(data_in)
            gts = data_in['gts'].cuda()
            # gts = self.rgb2y(gts)
            gts = gts.unsqueeze(2)
            scalar = self.params.validation_config.interp_ratio - 1
            n, _, _, h, w = gts.shape
            gts = gts.reshape(n, scalar, -1, h, w)
            # gts = torch.cat(gts.split(1, dim=1), dim=0)
            res_list = []
            
            for si in range(scalar):
                # for n in range(gts.shape[0]):
                recon_temp = self.forward(self.pack_valdata(data_in, si))[0]
                recon = (recon_temp[0]+recon_temp[1])/2.
                res_list.append(recon)
            recon = torch.stack(res_list, 1)
            if self.debug:
                self.update_validation_metrics(recon, gts, epoch, data_in, cache_dict=self.net.cache_dict)
            else:
                self.update_validation_metrics(recon, gts, epoch, data_in)
        return

    def forward(self, pack_data):
        res = self.net(pack_data, self.training_stage)
        if self.training_stage == 'flow':
            im = torch.stack((res[2], res[3]), 0)
            res_flow = torch.stack((res[0][0], res[1][0]), 0)
            return im, res_flow
        else:
            print("Not Implemented")
            exit()
