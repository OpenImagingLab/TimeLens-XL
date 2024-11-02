import pyiqa
from tools.registery import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class ssim():
    def __init__(self, loss_dict):
        self.loss_dict = loss_dict
        if 'weight' in loss_dict.keys():
            self.weight = loss_dict['weight']
            loss_dict.pop('weight')
        else:
            self.weight = 1.
        self.loss_item = pyiqa.create_metric('ssim', **loss_dict)

    def forward(self, x, y):
        return self.weight*self.loss_item(x.clamp(0., 1.), y.clamp(0., 1.))