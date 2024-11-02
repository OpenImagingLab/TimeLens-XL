import os
import cv2
import torch
import argparse
from torch.nn import functional as F
import warnings
from natsort import natsorted
warnings.filterwarnings("ignore")

mkdir = lambda x:os.makedirs(x, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
# parser.add_argument('--img', dest='img', nargs=2, required=True)
parser.add_argument('--folder', type=str, required=True)
parser.add_argument('--fdepth', type=int, default=2)
parser.add_argument('--exp', default=4, type=int)
parser.add_argument('--ratio', default=0, type=float, help='inference ratio between two images with 0 - 1 range')
parser.add_argument('--rthreshold', default=0.02, type=float, help='returns image when actual ratio falls in given range threshold')
parser.add_argument('--rmaxcycles', default=8, type=int, help='limit max number of bisectional cycles')
parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')
parser.add_argument('--output', type=str, default='./output')

args = parser.parse_args()



try:
    try:
        try:
            from model.RIFE_HDv2 import Model
            model = Model()
            model.load_model(args.modelDir, -1)
            print("Loaded v2.x HD model.")
        except:
            from train_log.RIFE_HDv3 import Model
            model = Model()
            model.load_model(args.modelDir, -1)
            print("Loaded v3.x HD model.")
    except:
        from model.RIFE_HD import Model
        model = Model()
        model.load_model(args.modelDir, -1)
        print("Loaded v1.x HD model")
except:
    from model.RIFE import Model
    model = Model()
    model.load_model(args.modelDir, -1)
    print("Loaded ArXiv-RIFE model")
model.eval()
model.device()

# if args.img[0].endswith('.exr') and args.img[1].endswith('.exr'):
#     img0 = cv2.imread(args.img[0], cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
#     img1 = cv2.imread(args.img[1], cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
#     img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device)).unsqueeze(0)
#     img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device)).unsqueeze(0)
#
# else:
#     img0 = cv2.imread(args.img[0], cv2.IMREAD_UNCHANGED)
#     img1 = cv2.imread(args.img[1], cv2.IMREAD_UNCHANGED)
#     img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
#     img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
#
# n, c, h, w = img0.shape
# ph = ((h - 1) // 32 + 1) * 32
# pw = ((w - 1) // 32 + 1) * 32
# padding = (0, pw - w, 0, ph - h)
# img0 = F.pad(img0, padding)
# img1 = F.pad(img1, padding)


def im_preprocessing(im_path):
    if im_path.endswith('.exr'):
        im = cv2.imread(im_path, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
        im = (torch.tensor(im.transpose(2, 0, 1)).to(device)).unsqueeze(0)
    else:
        im = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
        im = (torch.tensor(im.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    n, c, h, w = im.shape
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = (0, pw - w, 0, ph - h)
    im = F.pad(im, padding)
    return im, h, w

def parse_path(folder, fdepth):
    file_dict = {}
    if fdepth == 1:
        file_dict.update({
            os.path.split(folder)[-1]:[os.path.join(folder, file) for file in natsorted(os.listdir(folder))]
        })
    elif fdepth == 2:
        for subfolder in os.listdir(folder):
            sfolder = os.path.join(folder, subfolder)
            file_dict.update({
                subfolder:[os.path.join(sfolder, file) for file in natsorted(os.listdir(sfolder))]
            })
    else:
        for subfolder in os.listdir(folder):
            sfolder = os.path.join(folder, subfolder)
            for subsubfolder in os.listdir(sfolder):
                ssfolder = os.path.join(sfolder, subsubfolder)
                file_dict.update({
                    f"{subfolder}_{subsubfolder}":[os.path.join(ssfolder, file) for file in natsorted(os.listdir(ssfolder))]
                })
    return file_dict

file_dict = parse_path(args.folder, args.fdepth)
for ind, item in enumerate(file_dict.items()):
    k, v = item
    for fi in range(len(v)-1):
        img0, h, w = im_preprocessing(v[fi])
        img1 = im_preprocessing(v[fi+1])[0]
        if args.ratio:
            img_list = [img0]
            img0_ratio = 0.0
            img1_ratio = 1.0
            if args.ratio <= img0_ratio + args.rthreshold / 2:
                middle = img0
            elif args.ratio >= img1_ratio - args.rthreshold / 2:
                middle = img1
            else:
                tmp_img0 = img0
                tmp_img1 = img1
                for inference_cycle in range(args.rmaxcycles):
                    middle = model.inference(tmp_img0, tmp_img1)
                    middle_ratio = ( img0_ratio + img1_ratio ) / 2
                    if args.ratio - (args.rthreshold / 2) <= middle_ratio <= args.ratio + (args.rthreshold / 2):
                        break
                    if args.ratio > middle_ratio:
                        tmp_img0 = middle
                        img0_ratio = middle_ratio
                    else:
                        tmp_img1 = middle
                        img1_ratio = middle_ratio
            img_list.append(middle)
            img_list.append(img1)
        else:
            img_list = [img0, img1]
            for i in range(args.exp):
                tmp = []
                for j in range(len(img_list) - 1):
                    mid = model.inference(img_list[j], img_list[j + 1])
                    tmp.append(img_list[j])
                    tmp.append(mid)
                tmp.append(img1)
                img_list = tmp

        mkdir(args.output)
        mkdir(os.path.join(args.output, k))
        vind = os.path.splitext(os.path.split(v[fi])[-1])[0]
        cv2.imwrite('{}/{}.png'.format(os.path.join(args.output, k), vind),
                    (img_list[0][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])
        for i in range(1, len(img_list)-1):
            # if v[i].endswith('.exr') and v[i+1].endswith('.exr'):
            #     cv2.imwrite('{}/{}_{}.exr'.format(os.path.join(args.output, k), i),
            #                 (img_list[i][0]).cpu().numpy().transpose(1, 2, 0)[:h, :w], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
            # else:
            cv2.imwrite('{}/{}_{}.png'.format(os.path.join(args.output, k), vind, i), (img_list[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])

        vindNext = os.path.splitext(os.path.split(v[fi+1])[-1])[0]
        cv2.imwrite('{}/{}.png'.format(os.path.join(args.output, k), vindNext),
                    (img_list[len(img_list)-1][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])
