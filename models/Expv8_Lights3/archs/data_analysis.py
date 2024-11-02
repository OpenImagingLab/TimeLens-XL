from tools import toim
import os
import torch


target_path = 'vis_data'
mkdir = lambda x:os.makedirs(x, exist_ok=True)
mkdir(target_path)


def vis_events(e, ind):
	dm = max(torch.max(e).item(), torch.max(-e).item())
	print(e.max(), e.min())
	e = e/(2*dm)+0.5
	print(dm, e.max(), e.min())
	toim(e).save(os.path.join(target_path, f"{ind}.jpg"))