import torch
from tools.registery import LOSS_REGISTRY
from torch.nn import functional as F


@LOSS_REGISTRY.register()
class l1_loss():
    def __init__(self, loss_dict):
        self.loss_dict = loss_dict
        self.as_loss = loss_dict.as_loss
        self.weight = loss_dict.weight
        self.loss_item = torch.nn.L1Loss()

    def forward(self, x, y):
        if not self.as_loss:
            with torch.no_grad():
                return self.weight*self.loss_item(x, y)

        else:
            return self.weight*self.loss_item(x, y)


@LOSS_REGISTRY.register()
class psnr():
    def __init__(self, loss_dict):
        self.loss_dict = loss_dict
        self.as_loss = loss_dict.as_loss
        self.weight = loss_dict.weight
        self.loss_item = torch.nn.MSELoss()

    def forward(self, x, y, data_range=1.):
        if not self.as_loss:
            with torch.no_grad():
                mse = self.loss_item(x.clamp(0, 1.), y.clamp(0., 1.))
                return 10*torch.log10(data_range**2/(mse+1e-14))
        else:
            mse = self.loss_item(x, y)
            return 10*torch.log10(data_range**2/mse)

def charbonnier_loss(pred, target, eps=1e-6):
    return torch.mean(torch.sqrt((pred - target)**2 + eps))

@LOSS_REGISTRY.register()
class Charbonier():
    def __init__(self, loss_dict):
        self.loss_dict = loss_dict
        self.as_loss = loss_dict.as_loss
        self.weight = loss_dict.weight
        self.loss_item = charbonnier_loss

    def forward(self, x, y):
        if not self.as_loss:
            with torch.no_grad():
                loss = self.loss_item(x, y)
        else:
            loss = self.loss_item(x, y)
        return loss
    
@LOSS_REGISTRY.register()
class HalfCharbonier():
    def __init__(self, loss_dict):
        self.loss_dict = loss_dict
        self.as_loss = loss_dict.as_loss
        self.weight = loss_dict.weight
        self.loss_item = charbonnier_loss

    def forward(self, x, y):
        if len(x.shape) -- 5:
            N, T, C, H, W = x.shape
            x, y = F.interpolate(x.view(-1, C, H, W), scale_factor=0.5, mode='bilinear'), F.interpolate(y.view(-1, C, H, W), scale_factor=0.5, mode='bilinear')
        if not self.as_loss:
            with torch.no_grad():
                loss = self.loss_item(x, y)
        else:
            loss = self.loss_item(x, y)
        return loss
    
@LOSS_REGISTRY.register()
class GroupCharbonier():
    def __init__(self, loss_dict):
        self.loss_dict = loss_dict
        self.as_loss = loss_dict.as_loss
        self.weight = loss_dict.weight
        self.loss_item = charbonnier_loss

    def forward(self, xin, y):
        loss = 0.
        for idx in range(len(self.weight)):
            x = xin[idx]
            h, w = x.shape[2:]
            ycur = F.interpolate(y, (h, w), mode='bilinear')
            if not self.as_loss:
                with torch.no_grad():
                    loss += self.loss_item(x, ycur)*self.weight[idx]
            else:
                loss += self.loss_item(x, ycur)*self.weight[idx]
        return loss

@LOSS_REGISTRY.register()
class ECharbonier():
    def __init__(self, loss_dict):
        self.weight = loss_dict.weight
        self.loss_item = charbonnier_loss
        self.emask_type = loss_dict.emask_type

    def forward(self, x, y, e):
        e_mask = torch.sum(torch.abs(e), 1, keepdim=True)
        if self.emask_type == 'flag':
            e_mask = e_mask/torch.max(e_mask, torch.tensor(1e-5).to(e.device))
        loss = self.loss_item(x, y)
        loss_e = self.loss_item(x*e_mask, y*e_mask)*self.weight
        return loss+loss_e
    
@LOSS_REGISTRY.register()
class SmoothLoss():
    def __init__(self, loss_dict):
        self.weight = loss_dict.weight
        
    def gradients(self, img):
        dy = img[:,:,1:,:] - img[:,:,:-1,:]
        dx = img[:,:,:,1:] - img[:,:,:,:-1]
        return dx, dy
        
    def cal_grad2_error(self, flow, img):
        flow = flow / 20.
        img_grad_x, img_grad_y = self.gradients(img)
        w_x = torch.exp(-10.0 * torch.abs(img_grad_x).mean(1).unsqueeze(1))
        w_y = torch.exp(-10.0 * torch.abs(img_grad_y).mean(1).unsqueeze(1))

        dx, dy = self.gradients(flow)
        dx2, _ = self.gradients(dx)
        _, dy2 = self.gradients(dy)
        error = (w_x[:,:,:,1:] * torch.abs(dx2)).mean((1,2,3)) + (w_y[:,:,1:,:] * torch.abs(dy2)).mean((1,2,3))
        #error = (w_x * torch.abs(dx)).mean((1,2,3)) + (w_y * torch.abs(dy)).mean((1,2,3))
        return error / 2.0

    def forward(self, flow, gt):
        loss = self.weight*self.cal_grad2_error(flow, gt)
        return loss.mean()
    
    
@LOSS_REGISTRY.register()
class AnnealCharbonier():
    def __init__(self, loss_dict):
        self.loss_dict = loss_dict
        self.as_loss = loss_dict.as_loss
        self.weight = loss_dict.weight
        self.fact = loss_dict.fact
        self.loss_item = charbonnier_loss

    def forward(self, x, y, x0, step):
        if not self.as_loss:
            with torch.no_grad():
                loss0 = self.loss_item(x, y)
                loss1 = self.loss_item(x0, y)
        else:
            loss0 = self.loss_item(x, y)
            loss1 = self.loss_item(x0, y)
        curfact = self.fact**step
        loss = loss0*(1-curfact)+loss1*curfact
        return loss*self.weight