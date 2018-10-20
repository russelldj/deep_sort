import argparse
import os
import glob 
import pandas as pd
import numpy as np
import logging
import cv2
import time
import h5py

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
parser.add_argument('--output_dir', default=OUTPUT_DIR, type=str, help='where to dump the data, will be created if missing. The name of the tracks folder will be appended to this path' )
parser.add_argument('--MOT_style_images', action='store_true', default=False, help='All the images are in the form <VIDEO_ID>/img1/*' )
parser.add_argument('--conf_threshold', default=float("inf"), type=float, help="don't show detections with a confidence below this")
parser.add_argument('--visualize_frames_file', default=None, type=str, help="A file of frame indices which sould be visualized. One per line with no commas. If unset all will be visualized")
parser.add_argument('--visualize_fraction', type=float, default=1.0, help="This is the fraction of the frames which will be visulized. By chunks of 1000")
parser.add_argument('--track_type', type=str, default='txt', help="h5 or txt")
args = parser.parse_args()

CLASS_NAMES = ['__background__',
    'dent_floss', 'tap', 'shoes', 'laptop',\
            'shoe', 'keyboard', 'dish', 'tv_remote',\
            'pills', 'thermostat', 'knife/spoon/fork',\
            'tooth_paste', 'container', 'monitor',\
            'mug/cup', 'tv', 'kettle', 'microwave',\
            'cell', 'detergent', 'book', 'oven/stove',\
            'elec_keys', 'milk/juice', 'pan', 'tooth_brush',\
            'door', 'electric_keys', 'blanket', 'mop',\
            'pitcher', 'fridge', 'cloth', 'vacuum', 'comb',\
            'large_container', 'washer/dryer', 'towel',\
            'tea_bag', 'trash_can', 'food/snack', 'soap_liquid',\
            'bed', 'perfume', 'person', 'bottle', 'basket', 'cell_phone']

class Visualizer():
    def __init__(self):
        np.random.seed(0) # get a bit more consistency wrt to the colors
        self.colors = [(np.random.randint(0,256), np.random.randint(0,256), np.random.randint(0,256)) for _ in range(100000)] # generate a set of random numbers 

    def plot(self, img, tracks):
        """this function will modify the image inplace, rather than returning a copy.
        The tracks will be a pandas dataframe"""
        for index, track in tracks.iterrows():
            if track['conf'] < args.conf_threshold:
                continue # don't visualize this detection
            color = self.colors[int(track['ID'])]
            cv2.rectangle(img, (int(track['x']), int(track['y'])), (int(track['x'] + track['w']), int(track['y'] + track['h'])), color, LINE_WIDTH)
            USE_NAMES=False
            if USE_NAMES:
                cv2.putText(img, "{} {:01.3f}".format(CLASS_NAMES[int(track['ID'])], track['conf']), (int(track['x']),  int(track['y'] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
            else:
                cv2.putText(img, "{} {:01.3f}".format(str(int(track['ID'])), track['conf']), (int(track['x']),  int(track['y'] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


data_folders = sorted(glob.glob('{}/*'.format(args.image_dir)))
track_files = sorted(glob.glob('{}/*'.format(args.tracks_dir)))
print(data_folders)
print(track_files)
#assert len(data_folders) == len(track_files)

# make the output dir
output_folder = '{}/{}'.format(args.output_dir, args.tracks_dir.split('/')[-1]) # add the leaf folder of the tracks path
os.makedirs(output_folder, exist_ok=True)

CHUNK_LENGTH = 1000

for vid_ind, track_file in enumerate(track_files):
    visualizer = Visualizer()
    if args.track_type == 'txt':
        tracks = pd.read_csv(track_file, header = None, names = ["frame", "ID", "x", "y", "w", "h", "conf", "big_x", "big_y", "big_z"])
    elif args.track_type == 'h5':
        hf = h5py.File(track_file)
        data = hf.get('data')[:]
        tracks = pd.DataFrame(data=data, columns=["frame", "ID", "x", "y", "w", "h", "conf", "big_x", "big_y", "big_z"])
    else:
        raise ValueError('track_type wasn\'t txt or h5')

    tracks.sort_values(by=['frame'])
    #print(tracks)
    if args.MOT_style_images:
        images = sorted(glob.glob('{}/img1/*'.format(data_folders[vid_ind])))
    else:
        images = sorted(glob.glob('{}/*'.format(data_folders[vid_ind])))
    # make a folder to write the visualizations
    #TODO change this to create a video writer
    im_shape = cv2.imread(images[0]).shape

    video_fname = '{}/{}.avi'.format(output_folder, os.path.basename(track_file).split('.')[0])
    if os.path.isfile(video_fname):
        os.remove(video_fname)

    video_writer = cv2.VideoWriter(video_fname, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (im_shape[1], im_shape[0]))

    # read the indices to visualize
    if args.visualize_frames_file: # not None
        write_indices = np.loadtxt(args.visualize_frames_file, dtype=int)
    else:
        write_indices = np.arange(len(images), dtype=int)

    #MOD only look at the first few images 
    time_start = time.time()
    for img_ind, img_name in enumerate(images):
        if img_ind not in write_indices:
            continue
        PRINT_INTERVAL = 100
        if img_ind  % PRINT_INTERVAL == 0:
            print('elapsed time for {} to {} is {}.'.format(img_ind - PRINT_INTERVAL, img_ind, time.time() - time_start))
            time_start = time.time()

        if img_ind % int(CHUNK_LENGTH / args.visualize_fraction) > CHUNK_LENGTH:# this should be a floored int
            continue # skip visualizing a fraction of the videos

        img = cv2.imread(img_name)
        visualizer.plot(img, tracks[tracks.frame == img_ind])
        video_writer.write(img)
    #don't use close that might have been causing the issues
    video_writer.release()
