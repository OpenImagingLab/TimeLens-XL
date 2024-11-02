from flow_vis import flow_to_color
import torch
from torch.nn import Module
import losses
from tools.registery import LOSS_REGISTRY
from copy import deepcopy
import time
from torchvision.transforms import ToTensor, ToPILImage
import os
import numpy as np
from models.Forward_warp import ForwardWarp
import os
from flow_vis import flow_to_color
mkdir = lambda x:os.makedirs(x, exist_ok=True)


class BaseModel(Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.train_im_path = params.paths.save.train_im_path
        # self.val_im_path = params.paths.save.val_im_path

        self.val_im_path = os.path.join(params.paths.save.val_im_path, "images")
        self.val_flow_save = os.path.join(params.paths.save.val_im_path, "flow")
        self.val_flowvis_save = os.path.join(params.paths.save.val_im_path, "flow_vis")
        mkdir(self.val_im_path)
        mkdir(self.val_flow_save)
        mkdir(self.val_flowvis_save)
        self.record_txt = params.paths.save.record_txt
        self.val_record_txt = os.path.join(self.val_im_path, 'detailed_records')
        # self.metrics = metrics
        self.training_metrics = {}
        self.validation_metrics = {}
        self.train_print_freq = params.training_config.train_stats.print_freq
        self.train_im_save = params.training_config.train_stats.save_im_ep
        self.val_eval = params.validation_config.weights_save_freq
        self.val_im_save = params.validation_config.val_imsave_epochs
        self.interp_num = params.training_config.interp_ratio - 1
        self.toim = ToPILImage()
        self.metrics_init()
        # some assistance functions for optical flow
        self.fwarp = ForwardWarp()
        self.bwarp = self.backwarp
        self.params_training = None
        self.debug=params.debug
        self.save_flow = params.save_flow
        self.save_images = params.save_images


    def write_log(self, logcont):
        with open(self.record_txt, 'a+') as f:
            f.write(logcont)

    def _init_metrics(self, metrics):
        if metrics is None:
            metrics = {}
            for k in self.params.training_config.losses.keys():
                metrics.update({
                    f"train_{k}": []
                })
            for k in self.params.validation_config.losses.keys():
                metrics.update({
                    f"val_{k}": []
                })
        self.metrics = metrics
        self.metrics_record = deepcopy(metrics)
        self.metrics_record['training_time'] = []
        self.metrics_record['validation_time'] = []

    def metrics_init(self):
        for k in self.params.training_config.losses.keys():
            self.training_metrics.update({
                k:[LOSS_REGISTRY.get(k)(self.params.training_config.losses[k]),
                   self.params.training_config.losses[k]['as_loss']]
            })

        for k in self.params.validation_config.losses.keys():
            self.validation_metrics.update({
                k:LOSS_REGISTRY.get(k)(self.params.validation_config.losses[k])
            })

    def _update_training_time(self):
        self.metrics_record['training_time'].append(time.time())

    def _update_validation_time(self):
        self.metrics_record['validation_time'].append(time.time())

    def _reset_metrics_record(self):
        for k in self.metrics_record.keys():
            self.metrics_record[k] = []

    def _print_train_log(self, epoch):
        print_content = f"EPOCH/MAX EPOCH: {epoch}/{self.params.training_config.max_epoch}\t"
        for k in self.training_metrics.keys():
            print_content += f'{k}:{np.mean(self.metrics_record[f"train_{k}"]):.4f}\t'
        print_content = print_content.strip('\t') + '\n'
        print(print_content)
        return print_content

    def _print_val_log(self):
        print_content = f"Validation Logs: \t"
        for k in self.validation_metrics.keys():
            print_content += f'{k}:{np.mean(self.metrics_record[f"val_{k}"]):.4f}\t'
        print_content = print_content.strip('\t') + '\n'
        print(print_content)
        return print_content

    def backwarp(self, img, flow):
        _, _, H, W = img.size()

        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]

        gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))

        gridX = torch.tensor(gridX, requires_grad=False, ).cuda()
        gridY = torch.tensor(gridY, requires_grad=False, ).cuda()
        x = gridX.unsqueeze(0).expand_as(u).float() + u
        y = gridY.unsqueeze(0).expand_as(v).float() + v
        # range -1 to 1
        x = 2 * (x / W - 0.5)
        y = 2 * (y / H - 0.5)
        # stacking X and Y
        grid = torch.stack((x, y), dim=3)
        # Sample pixels using bilinear interpolation.
        imgOut = torch.nn.functional.grid_sample(img, grid)

        return imgOut

    def flow_resize(self, flow, target_h=None, target_w=None, scalar=None):
        fh, fw = flow.shape[-2:]
        if target_h is None:
            target_h, target_w = int(fh*scalar), int(fw*scalar)
        hr, wr = float(target_h) / float(fh), float(target_w) / float(fw)
        flow_out = torch.nn.functional.interpolate(flow, (target_h, target_w), mode='bilinear', align_corners=True)
        flow_out[:, 0] *= wr
        flow_out[:, 1] *= hr
        return flow_out

    def net_training(self, data_in, optim, epoch, step):
        pass

    def validation(self, data_in, epoch):
        pass

    def forward(self, *args, **kwargs):
        pass

    def update_training_metrics(self, res, gt, epoch, step, lr, *args, **kwargs):
        loss = 0
        print_content = f"MODEL {self.params.model_config.name}\tCur EPOCH/STEP/LR: [{epoch}/{step}/{lr:.6f}]\t"
        self.metrics_record["training_time"].append(time.time())
        if step % self.train_print_freq == 0:
            print_content += f'TIme: {self.metrics_record["training_time"][1]-self.metrics_record["training_time"][0]:.4f}\t' \
                if step == 0 else f'TIme: {self.metrics_record["training_time"][step]-self.metrics_record["training_time"][step-self.train_print_freq]:.4f}\t'
            if step == 0:
                self.metrics_record['training_time'].pop()

        for k in self.training_metrics.keys():
            func, as_loss = self.training_metrics[k]
            if k == 'ECharbonier':
                events =  args[0]
                loss_item = func.forward(res, gt, events)
            if k == "AnnealCharbonier":
                loss_item = func.forward(res, gt, kwargs['fuseout'], epoch)
            elif k == 'dists' or k=='lpips':
                if len(res.shape) == 5:
                    rn, rt, rc, rh, rw = res.shape
                    loss_item = func.forward(res.view(-1, rc, rh, rw), gt.view(-1, rc, rh, rw))
                else:
                    loss_item = func.forward(res, gt)
            else:
                loss_item = func.forward(res, gt)
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

    def update_validation_metrics(self, res, gt, epoch, data_in, *args, **kwargs):
        import os
        os.makedirs(self.val_record_txt, exist_ok=True)
        for n in range(res.shape[1]):
            detailed_record = f'EPOCH {epoch}\tFolder: {data_in["folder"][0]} Image: {data_in["rgb_name"][n]} val num: {n}\t'
            for k in self.validation_metrics.keys():
                val = self.validation_metrics[k].forward(res[:, n].detach(), gt[:, n].detach()).item()
                self.metrics_record[f'val_{k}'].append(val)
                detailed_record += f'{k}: {val:.4f}\t'
            with open(os.path.join(self.val_record_txt, f"{epoch}.txt"), 'a+') as f:
                f.write(detailed_record.strip('\t')+'\n')
            if (epoch % max(self.params.validation_config.val_imsave_epochs, 1) == 0 and self.save_images) or not self.params.enable_training:
                os.makedirs(os.path.join(self.val_im_path, str(epoch)), exist_ok=True)
                rgb_name = data_in['rgb_name']
                folder = os.path.split(data_in['folder'][0])[-1]
                # if n == 0:
                #     self.toim(data_in['im0'][0]).save(os.path.join(self.val_im_path, str(epoch), f"{folder}_{rgb_name[0][0]}_im0.jpg"))
                #     self.toim(data_in['im1'][0]).save(os.path.join(self.val_im_path, str(epoch), f"{folder}_{rgb_name[-1][0]}_im1.jpg"))
                self.toim(res[0, n].detach().cpu().clamp(0, 1)).save(os.path.join(self.val_im_path, str(epoch), f"{folder}_{rgb_name[n+1][0]}_{n}_res.jpg"))
                self.toim(gt[0, n]).save(os.path.join(self.val_im_path, str(epoch), f"{folder}_{rgb_name[n+1][0]}_{n}_gt.jpg"))
                # if self.debug:
                #     cache_dict = kwargs['cache_dict']
                #     for k in cache_dict.keys():
                #         v = cache_dict[k]
                #         if len(v) > 0:
                #             if k.startswith('Flow'):
                #                 self.toim(flow_to_color(v[n][0].permute(1, 2, 0).detach().cpu().numpy())).save(os.path.join(self.val_im_path, str(epoch), f"{folder}_{rgb_name[n+1][0]}_{n}_{k}.jpg"))
                #             else:
                #                 self.toim(v[n][0].detach().cpu()).save(os.path.join(self.val_im_path, str(epoch), f"{folder}_{rgb_name[n+1][0]}_{n}_{k}.jpg"))
                if self.save_flow:
                    if 'flow' not in kwargs.keys():
                        print("Flow saving is enabled but not provided! Plz have a check")
                        exit()
                    flow = kwargs['flow']
                    ft0, ft1 = flow
                    mkdir(os.path.join(self.val_flowvis_save, str(epoch)))
                    self.toim(flow_to_color(ft0[n][0].permute(1, 2, 0).detach().cpu().numpy())).save(
                        os.path.join(self.val_flowvis_save, str(epoch), f"{folder}_{rgb_name[n + 1][0]}_{n}_ft0.jpg"))
                    self.toim(flow_to_color(ft1[n][0].permute(1, 2, 0).detach().cpu().numpy())).save(
                        os.path.join(self.val_flowvis_save, str(epoch), f"{folder}_{rgb_name[n + 1][0]}_{n}_ft1.jpg"))

        if self.save_flow:
            if 'flow' not in kwargs.keys():
                print("Flow saving is enabled but not provided! Plz have a check")
                exit()
            flow = kwargs['flow']
            ft0, ft1 = flow
            ft0 = ft0.detach().cpu().numpy()
            ft1 = ft1.detach().cpu().numpy()
            mkdir(os.path.join(self.val_flow_save, str(epoch)))
            np.savez_compressed(os.path.join(self.val_flow_save, str(epoch), f"{folder}_{rgb_name[n+1][0]}_{n}_flow.npz"),
                                ft0=ft0,
                                ft1=ft1)

        return













