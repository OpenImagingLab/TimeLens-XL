import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), "models/raft/RAFT-master/core"))
print(sys.path)
import torch
from torch import nn
from raft import RAFT
from easydict import EasyDict as ED


class raft(nn.Module):
	def __init__(self):
		super().__init__()
		args = ED()
		args.model = './models/raft/raft-things.pth'
		args.mixed_precision = False
		args.alternate_corr = False
		args.small = False
		self.iters = 10
		model = RAFT(args)
		weights = torch.load(args.model)
		keys = weights.keys()
		for k in list(keys):
			new_key = f"model.{k.split('module.')[1]}"
			weights[new_key] = weights[k]
			weights.pop(k)
		self.model = model
		self.load_state_dict(weights)


	def forward(self, im0, im1):
		flow = self.model(im0, im1, iters=self.iters, test_mode=True)
		return flow
    
sys.path.pop(0)
