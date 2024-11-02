import shutil
import os
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, help='path of source video ')
parser.add_argument("--dest", type=str, help='path of dest video ')
parser.add_argument("--depth", type=int, default=1, help='folder depth')
parser.add_argument("--ds", type=int, default=1, help='times of down sample')
args = parser.parse_args()
# srun --partition=Pixel python ffmpeg_v2f.py --video=datasets/oppo-OSD1-v --dest=datasets/oppo-OSD1 --ds=1
#path to ffmpeg.exe
ffmpeg_dir=''
#video type
vtype = '.mp4'
if not os.path.exists(args.dest):
    os.mkdir(args.dest)
# else:
#     shutil.rmtree(args.dest)
#     os.mkdir(args.dest)

def extract_frames(video, outDir):
    error = ""
    print('{} -i {} -vsync 0 -qscale:v 2 {}/%06d.jpg'.format(os.path.join(ffmpeg_dir, "ffmpeg"), video, outDir))
    retn = os.system('{} -v quiet -i {} -vsync 0 -qscale:v 2 {}/%06d.jpg'.format(os.path.join(ffmpeg_dir, "ffmpeg"), video, outDir))
    if retn:
        error = "Error converting file:{}. Exiting.".format(video)
        sys.exit(error)
    counter = -1
    if args.ds is not 1:
        for i in sorted(os.listdir(outDir)):
            counter += 1
            if counter % args.ds == 0:
                continue
            os.remove(os.path.join(outDir, i))
    return error


if args.depth == 1:
    for i in os.listdir(args.video):
        print(i)
        extractionPath = os.path.join(args.dest,i.split('.')[0])
        if not os.path.exists(extractionPath):
            os.mkdir(extractionPath)
        extract_frames(os.path.join(args.video,i),extractionPath)
else:
    if not os.path.exists(os.path.join(args.dest,args.video.split('.')[0])):
        os.mkdir(os.path.join(args.dest,args.video.split('.')[0]))    
    extract_frames(args.video, os.path.join(args.dest,args.video.split('.')[0]))
    


