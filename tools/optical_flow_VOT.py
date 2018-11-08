import cv2
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from scipy import stats

"""The goal of this script is to test visual object tracking using optical flow
It should be interactivate and easy to use"""

class FlowTracker(object):
    def __init__(self, video_file, flow_dir, start_frame=1):
        self.video_reader = cv2.VideoCapture(video_file)
        self.video_reader.set(1,start_frame)
        self.flow_dir = flow_dir
        self.frame_idx = start_frame
        self.bbox = None
        self.x_hist = None
        self.y_hist = None
        self.fig, self.axs = plt.subplots(1, 2, tight_layout=True)
        pass

    def run(self):
        self.bbox = self.pick_location

    def predict(self, scale_factor=1.0):
        """
        params
        flow : np.array
            mxnx2, with the first channel representing the x displacement and the second y
        bounding_box : arraylike
            this should be in the form [l, t, r, b]
        """
        def fint(float_):
            return int(np.floor(float_))

        assert self.flow.shape[2] == 2, "the flow should be x, y displacement vectors"
        tracked_region = self.flow[fint(self.bbox[1]):fint(self.bbox[3]),fint(self.bbox[0]):fint(self.bbox[2])]
        x_size = fint(self.bbox[2]) - fint(self.bbox[0])
        y_size = fint(self.bbox[3]) - fint(self.bbox[1])
        


        print(tracked_region.shape)
        USE_MODE = True

        if USE_MODE:
            x_ave = stats.mode(tracked_region[...,0].astype(int), axis=None).mode[0] * scale_factor 
            y_ave = stats.mode(tracked_region[...,1].astype(int), axis=None).mode[0] * scale_factor
        else:  
            x_ave = np.average(tracked_region[...,0]) * scale_factor
            y_ave = np.average(tracked_region[...,1]) * scale_factor

        self.x_hist = np.histogram(tracked_region[...,0])
        self.y_hist = np.histogram(tracked_region[...,1])
       
        #todo move this to its own method
        
        x_ceoffs = np.polyfit(np.arange(x_size), tracked_region[int(y_size / 2), :, 0], 1)
        y_ceoffs = np.polyfit(np.arange(y_size), tracked_region[:, int(x_size/ 2), 1], 1)
        print("x_ave: {}, y_ave: {}".format(x_ave, y_ave))
        print("xslope: {}, yslope: {}".format(x_ceoffs[0] * x_size, y_ceoffs[0] * y_size))
        
        #WARNING this might be weird for python 2
        self.bbox[0] += x_ave - x_ceoffs[0]# * x_size / 2.0 
        self.bbox[2] += x_ave + x_ceoffs[0]# * x_size / 2.0
        self.bbox[1] += y_ave - y_ceoffs[0]# * y_size / 2.0
        self.bbox[3] += y_ave + y_ceoffs[0]# * y_size / 2.0


    def set_location_ltrb(self, bbox):
        self.bbox = bbox


    def pick_location(self):
        self.load_next()
        selection = cv2.selectROI(self.image)
        cv2.destroyAllWindows()
        self.bbox = [selection[0], selection[1], selection[0] + selection[2], selection[1] + selection[3]]
        print(self.bbox)


    def load_next(self):
        flow_prefix = "{}/{:06d}".format(self.flow_dir, self.frame_idx)
        self.flow = self.load_flow(flow_prefix)
        ret, self.image = self.video_reader.read()
        self.frame_idx += 1


    def load_flow(self, flow_prefix):
        #import pdb; pdb.set_trace()
                # there might be a missing scale factor
        x_flow = cv2.imread("{}_x.jpg".format(flow_prefix)) / 255.0 # they are both saved in the range 0-255
        y_flow = cv2.imread("{}_y.jpg".format(flow_prefix)) / 255.0
        minmax = pd.read_csv("{}_minmax.txt".format(flow_prefix), sep=":", names=["values"], index_col=0)

        xmin = minmax["values"]["xmin"]
        ymin = minmax["values"]["ymin"]
        xmax = minmax["values"]["xmax"]
        ymax = minmax["values"]["ymax"]

        x_range = xmax - xmin 
        y_range = ymax - ymin

        x_flow = x_flow * x_range + xmin
        y_flow = y_flow * y_range + ymin

        #TODO determine why this needs to be flipped like this
        return np.concatenate((-1 * x_flow[...,0:1], -1 * y_flow[...,0:1]), axis=2) # keep the dimensionality with 0:1

    def show_hist_and_image(self):
        def fint(float_):
            return int(np.floor(float_))

        assert self.flow.shape[2] == 2, "the flow should be x, y displacement vectors"
        tracked_region = self.flow[fint(self.bbox[1]):fint(self.bbox[3]),fint(self.bbox[0]):fint(self.bbox[2])]
        self.axs[0].clear()

        x_flow = tracked_region[...,0].flatten()
        y_flow = tracked_region[...,1].flatten() 
        self.axs[0].hist(x_flow, bins="auto")
        self.axs[0].hist(y_flow, bins="auto")

        flow_image = self.add_optical_flow(self.image, self.flow, 25, "Flow")
        self.axs[1].imshow(cv2.cvtColor(flow_image, cv2.COLOR_BGR2RGB))
        plt.pause(.05)


    def add_track(self, image=None):
        if image is None:
            image = self.image
        return cv2.rectangle(image,(int(self.bbox[0]),int(self.bbox[1])),(int(self.bbox[2]),int(self.bbox[3])),(0,255,0),3)
    
    def show_flow(self):
        self.add_optical_flow(self.image, self.flow, 25, "Flow")

    def add_optical_flow(self, Image,Flow,Divisor,name ):
        "Display image with a visualisation of a flow over the top. A divisor controls the density of the quiver plot."
        PictureShape = np.shape(Image)
        #determine number of quiver points there will be
        Imax = int(PictureShape[0]/Divisor)
        Jmax = int(PictureShape[1]/Divisor)
        #create a blank mask, on which lines will be drawn.
        mask = np.zeros_like(Image)
        for i in range(1, Imax):
            for j in range(1, Jmax):
                X1 = (i)*Divisor
                Y1 = (j)*Divisor
                X2 = int(X1 + Flow[X1,Y1,1])
                Y2 = int(Y1 + Flow[X1,Y1,0])
                X2 = np.clip(X2, 0, PictureShape[0])
                Y2 = np.clip(Y2, 0, PictureShape[1])
                #add all the lines to the mask
                mask = cv2.line(mask, (Y1,X1),(Y2,X2), [255, 255, 255], 1)
                mask = cv2.circle(mask, (Y2, X2), 1, (0, 255, 0), -1)

        #superpose lines onto image
        img = cv2.add(Image,mask)
        #print image
        return self.add_track(img)

FLOW_DIR = "/home/drussel1/data/ADL/flows"
VIDEO_FILE = "/home/drussel1/data/ADL/ADL_videos/P_18.MP4"

flow_tracker = FlowTracker(VIDEO_FILE, FLOW_DIR, int(sys.argv[1]))
flow_tracker.pick_location()
#flow_tracker.set_location_ltrb([594, 336, 676, 508])
while True:
    flow_tracker.load_next()
    flow_tracker.predict()
    #flow_tracker.show_track()
    flow_tracker.show_flow()
    flow_tracker.show_hist_and_image()
