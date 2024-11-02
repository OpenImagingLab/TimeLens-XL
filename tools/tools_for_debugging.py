from matplotlib.pyplot import imshow, show, figure
from torchvision.transforms import ToPILImage, ToTensor
import torch


toim = ToPILImage()
totensor = ToTensor()

def align_and_vis_images(im0, im1, res, gts):
	res = torch.cat(res[0].split(1), 2)
	gts = torch.cat(gts[0].split(1), 2)
	im  = torch.cat((im0, im1), 2)[0]
	outim = torch.cat((im, torch.cat((res, gts), 2)[0]))
	return toim(outim)