import cv2
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from scipy import stats
import pylab as pl

from roipoly.roipoly import roipoly

"""The goal of this script is to test visual object tracking using optical flow
It should be interactivate and easy to use"""

class FlowTracker(object):
    def __init__(self, video_file, flow_dir, start_frame=1, use_polygon=True):
        self.video_reader = cv2.VideoCapture(video_file)
        self.video_reader.set(1,start_frame)
        self.flow_dir = flow_dir
        self.frame_idx = start_frame
        self.bbox = None
        self.x_hist = None
        self.y_hist = None
        self.use_polygon = use_polygon
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        width  = self.video_reader.get(3)#cv2.CV_CAP_PROP_FRAME_WIDTH)   # float
        height = self.video_reader.get(4)#cv2.CV_CAP_PROP_FRAME_HEIGHT) # float
        FPS = 60.0
        OUTPUT_FILENAME = "output.avi"
        print(int(width), int(height))
        self.video_writer = cv2.VideoWriter(OUTPUT_FILENAME, fourcc, FPS, (int(width), int(height)))

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

        if self.use_polygon:
            self.polygon_predict(self.flow, self.polygon, scale_factor=2.0)
        else:
            self.box_predict(self.flow, self.bbox)
   

    def polygon_predict(self, flow, polygon, scale_factor=1.0):
        """ this function needs to only consider the points inside of the polygon
        They will initially be used for x and y averages, and then more complex tings"""# ting for Gabe
        x_points, y_points, num_masked_points = self.masked_flow(flow, polygon)

        x_flow = sum(sum(x_points)) / num_masked_points
        y_flow = sum(sum(y_points)) / num_masked_points
        polygon[...,0] += x_flow * scale_factor
        polygon[...,1] += y_flow * scale_factor
    
    def masked_flow(self, flow, polygon):
        print(polygon)
        mask = np.zeros_like(flow[...,0], dtype=np.uint8)
        cv2.fillPoly(mask, polygon.astype(np.int32), 1)
        # now its time to multiply the mask with the x and y flows and then devide my the number in the mask
        num_masked_points = float(sum(sum(mask)))  # sum is applied column(I think)-wise

        return (np.multiply(mask, flow[...,0]), np.multiply(mask, flow[...,1]) , num_masked_points)

    def box_predict(self, flow, bbox, scale_factor=1.0):
        def fint(float_):
            return int(np.floor(float_))

        tracked_region = self.flow[fint(self.bbox[1]):fint(self.bbox[3]),fint(self.bbox[0]):fint(self.bbox[2])]
        x_size = fint(self.bbox[2]) - fint(self.bbox[0])
        y_size = fint(self.bbox[3]) - fint(self.bbox[1])
        

        print(tracked_region.shape)
        use_mode = True

        if use_mode:
            x_ave = stats.mode(tracked_region[...,0].astype(int), axis=None).mode[0] * scale_factor 
            y_ave = stats.mode(tracked_region[...,1].astype(int), axis=None).mode[0] * scale_factor
        else:  
            x_ave = np.average(tracked_region[...,0]) * scale_factor
            y_ave = np.average(tracked_region[...,1]) * scale_factor

        self.x_hist = np.histogram(tracked_region[...,0])
        self.y_hist = np.histogram(tracked_region[...,1])
    
        #todo move this to its own method
        try:  
            x_ceoffs = np.polyfit(np.arange(x_size), tracked_region[int(y_size / 2), :, 0], 1)
            y_ceoffs = np.polyfit(np.arange(y_size), tracked_region[:, int(x_size/ 2), 1], 1)
        except ValueError:
            self.video_writer.release()
            print("The box went out of frame")
            exit()
        print("x_ave: {}, y_ave: {}".format(x_ave, y_ave))
        print("xslope: {}, yslope: {}".format(x_ceoffs[0] * x_size, y_ceoffs[0] * y_size))
        
        #warning this might be weird for python 2
        self.bbox[0] += x_ave #- x_ceoffs[0] * x_size / 2.0 
        self.bbox[2] += x_ave #+ x_ceoffs[0] * x_size / 2.0
        self.bbox[1] += y_ave #- y_ceoffs[0] * y_size / 2.0
        self.bbox[3] += y_ave #+ y_ceoffs[0] * y_size / 2.0


    def set_location_ltrb(self, bbox):
        self.bbox = bbox


    def pick_location(self):
        self.load_next()
        if self.use_polygon:
            pl.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
            ROI = roipoly(roicolor="r")
            self.polygon = ROI.getPoints() 
            self.polygon = np.asarray([np.asarray(self.polygon).transpose()], np.single)
        else:
            selection = cv2.selectROI(self.image)
            #plt.close('all')
            self.bbox = [selection[0], selection[1], selection[0] + selection[2], selection[1] + selection[3]]
        plt.close("all")

        self.fig, self.axs = plt.subplots(1, 2, tight_layout=True)

        cv2.destroyAllWindows()
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

        if self.use_polygon:
            x_flow, y_flow, _  = self.masked_flow(self.flow, self.polygon)
            print(x_flow.shape, y_flow.shape)
            x_flow = x_flow.flatten()
            y_flow = y_flow.flatten()

            x_flow = x_flow[np.nonzero(x_flow)] # note that nonzero returns indices
            y_flow = y_flow[np.nonzero(y_flow)]
        else:
            tracked_region = self.flow[fint(self.bbox[1]):fint(self.bbox[3]),fint(self.bbox[0]):fint(self.bbox[2])]

            x_flow = tracked_region[...,0].flatten()
            y_flow = tracked_region[...,1].flatten() 


        #import pdb;pdb.set_trace()
        self.axs[0].clear()
        self.axs[0].hist(x_flow, bins="auto", color="b")
        self.axs[0].hist(y_flow, bins="auto", color="r")

        flow_image = self.add_optical_flow(self.image, self.flow, 25, "Flow")
        
        self.axs[1].imshow(cv2.cvtColor(flow_image, cv2.COLOR_BGR2RGB))
        plt.pause(.05)
        self.video_writer.write(flow_image)


    def add_track(self, image=None):
        if image is None:
            image = self.image
        if self.use_polygon:
            print("adding polygon")
            return cv2.polylines(image, self.polygon.astype(np.int32), True,(0,255,0), 5)  
        else:
            return cv2.rectangle(image,(int(self.bbox[0]),int(self.bbox[1])),(int(self.bbox[2]),int(self.bbox[3])),(0,255,0),3)
    
    def show_flow(self):
        self.add_optical_flow(self.image, self.flow, 25, "Flow")
        cv2.imshow("image", self.image)
        cv2.waitKey(10)

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

    def standard_tracking(self):
        ret, self.image = self.video_reader.read()
        assert self.image is not None
        selection = cv2.selectROI(self.image)
        cv2.destroyAllWindows()
        while(self.image is not None):
            ret, self.image = self.video_reader.read()
            cv2.imshow("frame", self.image)
            cv2.waitKey(10)

FLOW_DIR = "/home/drussel1/data/ADL/flows"
VIDEO_FILE = "/home/drussel1/data/ADL/ADL_videos/P_18.MP4"

if len(sys.argv) > 2:
    flow_tracker = FlowTracker(VIDEO_FILE, FLOW_DIR, int(sys.argv[1]), False)
elif len(sys.argv) > 1:
    flow_tracker = FlowTracker(VIDEO_FILE, FLOW_DIR, int(sys.argv[1]))
else:
    flow_tracker = FlowTracker(VIDEO_FILE, FLOW_DIR)

flow_tracker.standard_tracking()
#
#flow_tracker.pick_location()
##flow_tracker.set_location_ltrb([594, 336, 676, 508])
#while True:
#    flow_tracker.load_next()
#    flow_tracker.predict()
#    #flow_tracker.show_track()
#    flow_tracker.add_track()
#    #flow_tracker.show_flow()
#    flow_tracker.show_hist_and_image()
