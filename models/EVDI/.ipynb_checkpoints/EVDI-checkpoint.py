#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ZhangX
"""
import torch
import torch.nn as nn
from .networks import ResnetBlock, get_norm_layer
import time
import numpy as np
from tools.registery import MODEL_REGISTRY
from models.BaseModel import BaseModel

class ChannelAttention(nn.Module):
    ## channel attention block
    def __init__(self, in_planes, ratio=16): 
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    ## spatial attention block
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

def conv_layer(inDim, outDim, ks, s, p, norm_layer='none'):
    ## convolutional layer
    conv = nn.Conv2d(inDim, outDim, kernel_size=ks, stride=s, padding=p)
    relu = nn.LeakyReLU(0.2, True)
    assert norm_layer in ('batch', 'instance', 'none')
    if norm_layer == 'none':
        seq = nn.Sequential(*[conv, relu])
    else:
        if (norm_layer == 'instance'):
            norm = nn.InstanceNorm2d(outDim, affine=False, track_running_stats=False) # instance norm
        else:
            momentum = 0.1
            norm = nn.BatchNorm2d(outDim, momentum = momentum, affine=True, track_running_stats=True)
        seq = nn.Sequential(*[conv, norm, relu])
    return seq

def LDI_subNet(inDim=32, outDim=1, norm='none'):  
    ## LDI network
    convBlock1 = conv_layer(inDim,64,3,1,1)
    convBlock2 = conv_layer(64,128,3,1,1,norm)
    convBlock3 = conv_layer(128,64,3,1,1,norm)
    convBlock4 = conv_layer(64,16,3,1,1,norm)
    conv = nn.Conv2d(16, outDim, 3, 1, 1) 
    seq = nn.Sequential(*[convBlock1, convBlock2, convBlock3, convBlock4, conv])
    return seq

def pre_subNet(inDim=128, outDim=16, norm='none', n_blocks = 2, para=[5,2,2]):
    # sub network in fusion
    pre_net = nn.ModuleList()
    for i in range(n_blocks-1):
        pre_layer = conv_layer(inDim,inDim*2,para[0],para[1],para[2],norm)
        pre_net.append(pre_layer)
        inDim = inDim * 2
    # last layer
    pre_layer = conv_layer(inDim,outDim,para[0],para[1],para[2],norm)
    pre_net.append(pre_layer)
    return pre_net

def post_subNet(inDim=128, outDim=16, norm='none', n_blocks = 2, para=[5,2,2]):
    # sub network in fusion
    post_net = nn.ModuleList()
    for i in range(n_blocks-1):
        post_layer = conv_layer(inDim,inDim//4, para[0],para[1],para[2],norm)
        post_net.append(post_layer)
        inDim = inDim // 2
    # last layer
    post_layer = conv_layer(inDim,outDim,para[0],para[1],para[2],norm)
    post_net.append(post_layer)
    return post_net

@MODEL_REGISTRY.register()
class EVDI_Net(BaseModel):
    def __init__(self, params):
        super(EVDI_Net, self).__init__(params)
        
        ## LDI network
        self.LDI = LDI_subNet(32,1)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid() 
        
        ## fusion network
        self.convBlock1 = conv_layer(5,16,3,1,1)
        self.Pre = pre_subNet(16,64,'none',n_blocks=2,para=[3,1,1])
        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()
        self.resBlock1 = ResnetBlock(64,'zero',get_norm_layer('none'), False, True)
        self.resBlock2 = ResnetBlock(64,'zero',get_norm_layer('none'), False, True)
        self.Post = post_subNet(128,16,'none',n_blocks=2, para=[3,1,1])
        self.conv = nn.Conv2d(16, 1, 3, 1, 1)

    def save_training_samples(self, res, gt, preevents, postevents, data_in, epoch, step):
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
                    self.toim(data_in['im0'][b][0]).save(os.path.join(save_folder, f"b{b}n{n}_im0_id{file_names[n][b]}.jpg"))
                elif n == N-1:
                    self.toim(data_in['im1'][b][0]).save(os.path.join(save_folder, f"b{b}n{n}_im1_id{file_names[n][b]}.jpg"))
                else:
                    self.toim((res[(n-1)*B+b]).clamp(0, 1)).save(os.path.join(save_folder, f"b{b}n{n}_res_id{file_names[n][b]}.jpg"))
                    self.toim(gt[(n-1) * B + b]).save(os.path.join(save_folder, f"b{b}n{n}_gt_id{file_names[n][b]}.jpg"))
        return

    def update_training_metrics(self, res, gt, epoch, step, lr):
        loss = 0
        print_content = f"MODEL EVDI_Net\tCur EPOCH/STEP/LR: [{epoch}/{step}/{lr:.6f}]\t"
        self.metrics_record["training_time"].append(time.time())
        if step % self.train_print_freq == 0:
            print_content += f'TIme: {self.metrics_record["training_time"][1]-self.metrics_record["training_time"][0]:.4f}\t' \
                if step == 0 else f'TIme: {self.metrics_record["training_time"][step]-self.metrics_record["training_time"][step-self.train_print_freq]:.4f}\t'
            if step == 0:
                self.metrics_record['training_time'].pop()

        for k in self.training_metrics.keys():
            func, as_loss = self.training_metrics[k]
            loss_item = func.forward(res, gt)
            if as_loss:
                loss += loss_item
            self.metrics_record[f"train_{k}"].append(loss_item.item())
            print_content += f'{k}: {loss_item.item():.6f}\t'
            if step % self.train_print_freq:
                print_content += f'{k}: {self.metrics_record[f"train_{k}"][-1]:.4f}\t' \
                    if step == 0 else f'{k}: {np.mean(self.metrics_record[f"train_{k}"][step-self.train_print_freq:step]):.4f}\t'
        if step % self.train_print_freq == 0:
            print(print_content)
        return loss

    def net_training(self, data_in, optim, epoch, step):
        self.train()
        optim.zero_grad()
        print(data_in.keys())
        left_frame, right_frame, preevents, leftB_coef = data_in['im0'].cuda(), \
            data_in['im1'].cuda(), data_in['previous_events'].cuda(), data_in['left_weight']
        postevents = data_in['post_events'].cuda()
        leftB_coef = torch.stack(leftB_coef).float().permute(1, 0).unsqueeze(2).unsqueeze(3).cuda().split(1, 1)
        leftB_coef = torch.cat(leftB_coef, 0)
        rightB_coef = 1-leftB_coef
        gts = data_in['gts'].cuda()
        gts = torch.cat(gts.split(1, dim=1), dim=0)
        left_frame = torch.cat(left_frame.split(1, dim=1), dim=0)
        right_frame = torch.cat(right_frame.split(1, dim=1), dim=0)
        preevents = torch.cat(preevents.split(self.params.model_config.events_channel_num*2, dim=1), dim=0)
        postevents = torch.cat(postevents.split(self.params.model_config.events_channel_num*2, dim=1), dim=0)

        recon = self.forward(left_frame, right_frame, preevents, postevents, leftB_coef, rightB_coef)[0]
        loss = self.update_training_metrics(recon, gts, epoch, step, optim.param_groups[0]['lr'])
        loss.backward()
        optim.step()

        if epoch % self.train_im_save == 0 and step % self.train_print_freq == 0:
            self.save_training_samples(recon, gts, preevents, postevents, data_in, epoch, step)
        return

    def update_validation_metrics(self, res, gt, epoch, data_in, n):
        import os
        os.makedirs(self.val_record_txt, exist_ok=True)
        detailed_record = f'EPOCH {epoch}\tFolder: {data_in["folder"][0]} Image: {data_in["rgb_name"][n]} val num: {n}\t'
        for k in self.validation_metrics.keys():
            val = self.validation_metrics[k].forward(res.detach(), gt.detach()).item()
            self.metrics_record[f'val_{k}'].append(val)
            detailed_record += f'{k}: {val:.4f}\t'
        with open(os.path.join(self.val_record_txt, f"{epoch}.txt"), 'a+') as f:
            f.write(detailed_record.strip('\t')+'\n')
        if epoch % self.params.validation_config.val_imsave_epochs == 0:
            os.makedirs(os.path.join(self.val_im_path, str(epoch)), exist_ok=True)
            rgb_name = data_in['rgb_name']
            folder = data_in['folder'][0]
            if n == 1:
                self.toim(data_in['im0'][0]).save(os.path.join(self.val_im_path, str(epoch), f"{folder}_{rgb_name[0][0]}_im0.jpg"))
                self.toim(data_in['im1'][0]).save(os.path.join(self.val_im_path, str(epoch), f"{folder}_{rgb_name[-1][0]}_im1.jpg"))

            self.toim(res[0].detach().cpu()).save(os.path.join(self.val_im_path, str(epoch), f"{folder}_{rgb_name[n][0]}_{n}_res.jpg"))
            self.toim(gt[0]).save(os.path.join(self.val_im_path, str(epoch), f"{folder}_{rgb_name[n][0]}_{n}_gt.jpg"))
        return


    def net_validation(self, data_in, epoch):
        self.eval()
        with torch.no_grad():
            left_frame, right_frame, preevents, leftB_coef = data_in['im0'].cuda(), \
                data_in['im1'].cuda(), data_in['previous_events'].cuda(), data_in['left_weight']
            postevents = data_in['post_events'].cuda()
            leftB_coef = torch.stack(leftB_coef).float().permute(1, 0).unsqueeze(2).unsqueeze(3).cuda().split(1, 1)
            leftB_coef = torch.cat(leftB_coef, 0)
            rightB_coef = 1-leftB_coef
            gts = data_in['gts'].cuda()
            gts = torch.cat(gts.split(1, dim=1), dim=0)
            left_frame = torch.cat(left_frame.split(1, dim=1), dim=0)
            right_frame = torch.cat(right_frame.split(1, dim=1), dim=0)
            preevents = torch.cat(preevents.split(self.params.model_config.events_channel_num*2, dim=1), dim=0)
            postevents = torch.cat(postevents.split(self.params.model_config.events_channel_num*2, dim=1), dim=0)

            for n in range(gts.shape[0]):
                recon = self.forward(left_frame[n:n+1],
                                     right_frame[n:n+1],
                                     preevents[n:n+1],
                                     postevents[n:n+1],
                                     leftB_coef[n:n+1],
                                     rightB_coef[n:n+1],
                                     val=True
                                    )[0]
                self.update_validation_metrics(recon, gts[n:n+1], epoch, data_in, n+1)
        return

    def forward(self, left_frame, right_frame, previous_events, post_events, leftB_coef, rightB_coef, val=False):
        '''
        Parameters
        ----------
        leftB : left blurry image.
        rightB : left blurry image.
        leftB_inp1 : first event segment for leftB.
        leftB_inp2 : second event segment for leftB.
        leftB_w1 : weight for first event segment (related to leftB).
        leftB_w2 : weight for second event segment (related to leftB).
        rightB_inp1 : first event segment for rightB.
        rightB_inp2 : second event segment for rightB.
        rightB_w1 : weight for first event segment (related to rightB).
        rightB_w2 : weight for second event segment (related to rightB).
        leftB_coef : coefficient for L^i_(i+1), i.e., \omega in paper.
        rightB_coef : coefficient for L^i_(i+1), i.e., 1-\omega in paper.

        Returns
        -------
        recon : final reconstruction result.
        Ef1 : learned double integral of events (related to leftB).
        Ef2 : learned double integral of events (related to rightB).
        '''

        ## process by LDI networks
        Ef1_tmp1 = self.LDI(previous_events)
        Ef1 = self.relu(Ef1_tmp1) + self.sigmoid(Ef1_tmp1)

        Ef2_tmp1 = self.LDI(post_events)
        Ef2 = self.relu(Ef2_tmp1) + self.sigmoid(Ef2_tmp1)

        ## process by fusion network
        # generate recon3
        if not val:
            B,C,H,W = left_frame.shape
            N = Ef1.shape[0] // B
            Ef1 = Ef1.reshape((B,N,C,H,W))
            Ef2 = Ef2.reshape((B,N,C,H,W))
            leftB = left_frame.unsqueeze(1).repeat(1,N,1,1,1)
            rightB = right_frame.unsqueeze(1).repeat(1,N,1,1,1)
            recon1 = leftB * Ef1
            recon2 = rightB / Ef2
            recon1 = recon1.reshape((B*N,C,H,W))
            recon2 = recon2.reshape((B*N,C,H,W))
            leftB = leftB.reshape((B*N,C,H,W))
            rightB = rightB.reshape((B*N,C,H,W))
            Ef1 = Ef1.reshape((B*N,C,H,W))
            Ef2 = Ef2.reshape((B*N,C,H,W))
        else:
            recon1 = left_frame * Ef1
            recon2 = right_frame / Ef2
        recon3 = recon1 * leftB_coef + recon2 * rightB_coef

        # generate final result
        x = torch.cat((recon1,recon2,recon3,Ef1,Ef2), 1)
        x = self.convBlock1(x)
        blocks = []
        for i, pre_layer in enumerate(self.Pre):
            x = pre_layer(x)
            blocks.append(x)
        x = self.resBlock1(x)
        x = self.resBlock2(x)
        x = self.ca(x) * x
        x = self.sa(x) * x
        for i, post_layer in enumerate(self.Post):
            x = torch.cat((x, blocks[len(blocks)-i-1]), 1)
            x = post_layer(x)
        x = self.conv(x)
        recon = self.sigmoid(x)

        return recon, Ef1, Ef2

@MODEL_REGISTRY.register()
class EVDI_Color_Net(BaseModel):
    def __init__(self, params):
        super().__init__(params)
        
        ## LDI network
        self.LDI = LDI_subNet(32,3)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid() 
        
        ## fusion network
        self.convBlock1 = conv_layer(15,16,3,1,1)
        self.Pre = pre_subNet(16,64,'none',n_blocks=2,para=[3,1,1])
        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()
        self.resBlock1 = ResnetBlock(64,'zero',get_norm_layer('none'), False, True)
        self.resBlock2 = ResnetBlock(64,'zero',get_norm_layer('none'), False, True)
        self.Post = post_subNet(128,16,'none',n_blocks=2, para=[3,1,1])
        self.conv = nn.Conv2d(16, 3, 3, 1, 1)


    def save_training_samples(self, res, gt, preevents, postevents, data_in, epoch, step):
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
                    self.toim(data_in['im0'][b][0]).save(os.path.join(save_folder, f"b{b}n{n}_im0_id{file_names[n][b]}.jpg"))
                elif n == N-1:
                    self.toim(data_in['im1'][b][0]).save(os.path.join(save_folder, f"b{b}n{n}_im1_id{file_names[n][b]}.jpg"))
                else:
                    self.toim((res[(n-1)*B+b]).clamp(0, 1)).save(os.path.join(save_folder, f"b{b}n{n}_res_id{file_names[n][b]}.jpg"))
                    self.toim(gt[(n-1) * B + b]).save(os.path.join(save_folder, f"b{b}n{n}_gt_id{file_names[n][b]}.jpg"))
        return


    def net_training(self, data_in, optim, epoch, step):
        self.train()
        optim.zero_grad()
        left_frame, right_frame, preevents, leftB_coef = data_in['im0'].cuda(), \
            data_in['im1'].cuda(), data_in['previous_events'].cuda(), data_in['left_weight']
        postevents = data_in['post_events'].cuda()
        leftB_coef = torch.stack(leftB_coef).float().permute(1, 0).unsqueeze(2).unsqueeze(3).cuda().split(1, 1)
        leftB_coef = torch.cat(leftB_coef, 0)
        rightB_coef = 1-leftB_coef
        gts = data_in['gts'].cuda()
        gts = torch.cat(gts.split(3, dim=1), dim=0)
        left_frame = torch.cat(left_frame.split(3, dim=1), dim=0)
        right_frame = torch.cat(right_frame.split(3, dim=1), dim=0)
        preevents = torch.cat(preevents.split(self.params.model_config.events_channel_num*2, dim=1), dim=0)
        postevents = torch.cat(postevents.split(self.params.model_config.events_channel_num*2, dim=1), dim=0)

        recon = self.forward(left_frame, right_frame, preevents, postevents, leftB_coef, rightB_coef)[0]
        loss = self.update_training_metrics(recon, gts, epoch, step, optim.param_groups[0]['lr'])
        loss.backward()
        optim.step()

        if epoch % self.train_im_save == 0 and step % self.train_print_freq == 0:
            self.save_training_samples(recon, gts, preevents, postevents, data_in, epoch, step)
        return

    def net_validation(self, data_in, epoch):
        self.eval()
        with torch.no_grad():
            left_frame, right_frame, preevents, leftB_coef = data_in['im0'].cuda(), \
                data_in['im1'].cuda(), data_in['previous_events'].cuda(), data_in['left_weight']
            postevents = data_in['post_events'].cuda()
            leftB_coef = torch.stack(leftB_coef).float().permute(1, 0).unsqueeze(2).unsqueeze(3).cuda().split(1, 1)
            leftB_coef = torch.cat(leftB_coef, 0)
            rightB_coef = 1-leftB_coef
            gts = data_in['gts'].cuda()
            gts = torch.cat(gts.split(3, dim=1), dim=0)
            left_frame = torch.cat(left_frame.split(3, dim=1), dim=0)
            right_frame = torch.cat(right_frame.split(3, dim=1), dim=0)
            preevents = torch.cat(preevents.split(self.params.model_config.events_channel_num*2, dim=1), dim=0)
            postevents = torch.cat(postevents.split(self.params.model_config.events_channel_num*2, dim=1), dim=0)

            for n in range(gts.shape[0]):
                recon = self.forward(left_frame[n:n+1],
                                     right_frame[n:n+1],
                                     preevents[n:n+1],
                                     postevents[n:n+1],
                                     leftB_coef[n:n+1],
                                     rightB_coef[n:n+1],
                                     val=True
                                    )[0]
                self.update_validation_metrics(recon, gts[n:n+1], epoch, data_in, n+1)
        return

    # def forward(self, leftB_inp1, leftB_inp2, leftB_w1, leftB_w2,
    #             rightB_inp1, rightB_inp2, rightB_w1, rightB_w2,
    #             leftB, rightB, leftB_coef, rightB_coef):
    #     '''
    #     Parameters
    #     ----------
    #     leftB : left blurry image.
    #     rightB : left blurry image.
    #     leftB_inp1 : first event segment for leftB.
    #     leftB_inp2 : second event segment for leftB.
    #     leftB_w1 : weight for first event segment (related to leftB).
    #     leftB_w2 : weight for second event segment (related to leftB).
    #     rightB_inp1 : first event segment for rightB.
    #     rightB_inp2 : second event segment for rightB.
    #     rightB_w1 : weight for first event segment (related to rightB).
    #     rightB_w2 : weight for second event segment (related to rightB).
    #     leftB_coef : coefficient for L^i_(i+1), i.e., \omega in paper.
    #     rightB_coef : coefficient for L^i_(i+1), i.e., 1-\omega in paper.
    #
    #     Returns
    #     -------
    #     recon : final reconstruction result.
    #     Ef1 : learned double integral of events (related to leftB).
    #     Ef2 : learned double integral of events (related to rightB).
    #     '''
    #
    #     ## process by LDI networks
    #     Ef1_tmp1 = self.LDI(leftB_inp1)
    #     Ef1_tmp2 = self.LDI(leftB_inp2)
    #     Ef1 = leftB_w1 * Ef1_tmp1 + leftB_w2 * Ef1_tmp2
    #     Ef1 = self.relu(Ef1) + self.sigmoid(Ef1)
    #
    #     Ef2_tmp1 = self.LDI(rightB_inp1)
    #     Ef2_tmp2 = self.LDI(rightB_inp2)
    #     Ef2 = rightB_w1 * Ef2_tmp1 + rightB_w2 * Ef2_tmp2
    #     Ef2 = self.relu(Ef2) + self.sigmoid(Ef2)
    #
    #     ## process by fusion network
    #     # generate recon3
    #     B,C,H,W = leftB.shape
    #     N = Ef1.shape[0] // B
    #     Ef1 = Ef1.reshape((B,N,C,H,W))
    #     Ef2 = Ef2.reshape((B,N,C,H,W))
    #     leftB = leftB.unsqueeze(1).repeat(1,N,1,1,1)
    #     rightB = rightB.unsqueeze(1).repeat(1,N,1,1,1)
    #     recon1 = leftB / Ef1
    #     recon2 = rightB / Ef2
    #     recon1 = recon1.reshape((B*N,C,H,W))
    #     recon2 = recon2.reshape((B*N,C,H,W))
    #     leftB = leftB.reshape((B*N,C,H,W))
    #     rightB = rightB.reshape((B*N,C,H,W))
    #     Ef1 = Ef1.reshape((B*N,C,H,W))
    #     Ef2 = Ef2.reshape((B*N,C,H,W))
    #     recon3 = recon1 * leftB_coef + recon2 * rightB_coef
    #
    #     # generate final result
    #     x = torch.cat((recon1,recon2,recon3,Ef1,Ef2), 1)
    #     x = self.convBlock1(x)
    #     blocks = []
    #     for i, pre_layer in enumerate(self.Pre):
    #         x = pre_layer(x)
    #         blocks.append(x)
    #     x = self.resBlock1(x)
    #     x = self.resBlock2(x)
    #     x = self.ca(x) * x
    #     x = self.sa(x) * x
    #     for i, post_layer in enumerate(self.Post):
    #         x = torch.cat((x, blocks[len(blocks)-i-1]), 1)
    #         x = post_layer(x)
    #     x = self.conv(x)
    #     recon = self.sigmoid(x)
    #
    #     return recon, Ef1, Ef2
    def forward(self, left_frame, right_frame, previous_events, post_events, leftB_coef, rightB_coef, val=False):
        '''
        Parameters
        ----------
        leftB : left blurry image.
        rightB : left blurry image.
        leftB_inp1 : first event segment for leftB.
        leftB_inp2 : second event segment for leftB.
        leftB_w1 : weight for first event segment (related to leftB).
        leftB_w2 : weight for second event segment (related to leftB).
        rightB_inp1 : first event segment for rightB.
        rightB_inp2 : second event segment for rightB.
        rightB_w1 : weight for first event segment (related to rightB).
        rightB_w2 : weight for second event segment (related to rightB).
        leftB_coef : coefficient for L^i_(i+1), i.e., \omega in paper.
        rightB_coef : coefficient for L^i_(i+1), i.e., 1-\omega in paper.

        Returns
        -------
        recon : final reconstruction result.
        Ef1 : learned double integral of events (related to leftB).
        Ef2 : learned double integral of events (related to rightB).
        '''

        ## process by LDI networks
        Ef1_tmp1 = self.LDI(previous_events)
        Ef1 = self.relu(Ef1_tmp1) + self.sigmoid(Ef1_tmp1)

        Ef2_tmp1 = self.LDI(post_events)
        Ef2 = self.relu(Ef2_tmp1) + self.sigmoid(Ef2_tmp1)

        ## process by fusion network
        # generate recon3
        if not val:
            B,C,H,W = left_frame.shape
            N = Ef1.shape[0] // B
            Ef1 = Ef1.reshape((B,N,C,H,W))
            Ef2 = Ef2.reshape((B,N,C,H,W))
            leftB = left_frame.unsqueeze(1).repeat(1,N,1,1,1)
            rightB = right_frame.unsqueeze(1).repeat(1,N,1,1,1)
            recon1 = leftB * Ef1
            recon2 = rightB / Ef2
            recon1 = recon1.reshape((B*N,C,H,W))
            recon2 = recon2.reshape((B*N,C,H,W))
            leftB = leftB.reshape((B*N,C,H,W))
            rightB = rightB.reshape((B*N,C,H,W))
            Ef1 = Ef1.reshape((B*N,C,H,W))
            Ef2 = Ef2.reshape((B*N,C,H,W))
        else:
            recon1 = left_frame * Ef1
            recon2 = right_frame / Ef2
        recon3 = recon1 * leftB_coef + recon2 * rightB_coef

        # generate final result
        x = torch.cat((recon1,recon2,recon3,Ef1,Ef2), 1)
        x = self.convBlock1(x)
        blocks = []
        for i, pre_layer in enumerate(self.Pre):
            x = pre_layer(x)
            blocks.append(x)
        x = self.resBlock1(x)
        x = self.resBlock2(x)
        x = self.ca(x) * x
        x = self.sa(x) * x
        for i, post_layer in enumerate(self.Post):
            x = torch.cat((x, blocks[len(blocks)-i-1]), 1)
            x = post_layer(x)
        x = self.conv(x)
        recon = self.sigmoid(x)

        return recon, Ef1, Ef2
