import os
from natsort import natsorted as sorted


path = '/mnt/workspace/mayongrui/dataset/vimeo_septuplet/sequences/'
target_path = '/mnt/workspace/mayongrui/dataset/vimeo_septuplet/interpx8sequences/'
os.makedirs(target_path, exist_ok=True)
folders = os.listdir(path)
fl = len(folders)
for folder in folders[fl*3//4:]:
    subpath = os.path.join(path, folder)
    subtargetpath = os.path.join(target_path, folder)
    os.makedirs(subtargetpath, exist_ok=True)
    print(f"Processing: {folder}")
    cmd = f"CUDA_VISIBLE_DEVICES=0 python inference_folderims.py --folder {subpath} --exp 3 --output {subtargetpath}"
    print(cmd)
    os.system(cmd)
