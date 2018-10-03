import argparse
import os
import glob 
import pandas as pd
import numpy as np
import logging
import cv2
import time

XMIN = 0
YMIN = 1
XMAX = 2
YMAX = 3
ID_LOCATION = 4
LINE_WIDTH = 5
DATA_DIR = "./MOT16/custom_80" # where the images are 
#should be structured like DATA_DIR/<video id>/img1, DATA_DIR/<video id>/img1
TRACKS_DIR = "./custom/custom_results/conf_.65" # where the track results are
OUTPUT_DIR = '/home/drussel1/dev/deep_sort/custom/visualized_outputs/conf_.4'

parser = argparse.ArgumentParser()
parser.add_argument('--tracks_dir', default=TRACKS_DIR, type=str, help='where are the tracks that need to be visualized.' )
parser.add_argument('--image_dir', default=DATA_DIR, type=str, help='where are the videos that need to be visualized.') 
parser.add_argument('--output_dir', default=OUTPUT_DIR, type=str, help='where to dump the data, will be created if missing.' )
parser.add_argument('--MOT_style_images', action='store_true', default=False, help='All the images are in the form <VIDEO_ID>/img1/*' )
args = parser.parse_args()

class Visualizer():
    def __init__(self):
        np.random.seed(0) # get a bit more consistency wrt to the colors
        self.colors = [(np.random.randint(0,256), np.random.randint(0,256), np.random.randint(0,256)) for _ in range(100000)] # generate a set of random numbers 

    def plot(self, img, tracks):
        """this function will modify the image inplace, rather than returning a copy.
        The tracks will be a pandas dataframe"""
        for index, track in tracks.iterrows():
            color = self.colors[int(track['ID'])]
            cv2.rectangle(img, (int(track['x']), int(track['y'])), (int(track['x'] + track['w']), int(track['y'] + track['h'])), color, LINE_WIDTH)
            cv2.putText(img, str(int(track['ID'])), (int(track['x']),  int(track['y'] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)



data_folders = sorted(glob.glob('{}/*'.format(args.image_dir)))
track_files = sorted(glob.glob('{}/*'.format(args.tracks_dir)))
print(data_folders)
print(track_files)
assert len(data_folders) == len(track_files)

# make the output dir
os.makedirs(args.output_dir, exist_ok=True)

for vid_ind, track_file in enumerate(track_files):
    visualizer = Visualizer()
    tracks = pd.read_csv(track_file, header = None, names = ["frame", "ID", "x", "y", "w", "h", "conf", "big_x", "big_y", "big_z"])
    tracks.sort_values(by=['frame'])
    #print(tracks)
    if args.MOT_style_images:
        images = sorted(glob.glob('{}/img1/*'.format(data_folders[vid_ind])))
    else:
        images = sorted(glob.glob('{}/*'.format(data_folders[vid_ind])))
    # make a folder to write the visualizations
    os.makedirs('{}/{}'.format(args.output_dir, os.path.basename(track_file).split('.')[0]), exist_ok=True)
    for img_ind, img_name in enumerate(images):
        img = cv2.imread(img_name)
        visualizer.plot(img, tracks[tracks.frame == img_ind])
        cv2.imwrite('{}/{}/{:06d}.jpeg'.format(args.output_dir, os.path.basename(track_file).split('.')[0], img_ind), img)
