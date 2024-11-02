import torch


def print_grad(net):
	for p in net.named_parameters():
		name, variable = p[0], p[1]
		if variable.requires_grad and variable.grad is not None:
			print('+'*20, name, variable.requires_grad, variable.grad.max().item(), variable.grad.min().item())
		else:
			print('!'*20, name, variable.requires_grad)
