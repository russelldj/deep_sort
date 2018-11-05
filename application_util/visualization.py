# vim: expandtab:ts=4:sw=4
import numpy as np
import colorsys
from .image_viewer import ImageViewer
import pdb
import random


def create_unique_color_float(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def create_unique_color_uchar(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]

    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255*r), int(255*g), int(255*b)


class NoVisualization(object):
    """
    A dummy visualization object that loops through all frames in a given
    sequence to update the tracker without performing any visualization.
    """

    def __init__(self, seq_info):
        self.frame_idx = seq_info["min_frame_idx"]
        self.last_idx = seq_info["max_frame_idx"]

    def set_image(self, image):
        pass

    def draw_groundtruth(self, track_ids, boxes):
        pass

    def draw_detections(self, detections):
        pass

    def draw_trackers(self, trackers):
        pass

    def run(self, frame_callback, good_frames):
        while self.frame_idx <= self.last_idx:
            print(self.frame_idx)
            if good_frames is not None and self.frame_idx not in good_frames: #this should short circuit
                self.frame_idx += 1
                continue
            frame_callback(self, self.frame_idx)
            self.frame_idx += 1


class Visualization(object):
    """
    This class shows tracking output in an OpenCV image viewer.
    """

    def __init__(self, seq_info, update_ms):
        image_shape = seq_info["image_size"][::-1]
        aspect_ratio = float(image_shape[1]) / image_shape[0]
        image_shape = 1024, int(aspect_ratio * 1024)
        self.viewer = ImageViewer(
            update_ms, image_shape, "Figure %s" % seq_info["sequence_name"])
        self.viewer.thickness = 2
        self.index_to_vis = 1
        #MOD
        #HACK
        #added vid
        #self.viewer.enable_videowriter("/home/drussel1/dev/deep_sort/outputs/EPIC/visualizations/new_alg/no_new_alg.avi", fps=10)
        #self.viewer.enable_videowriter("/home/drussel1/dev/deep_sort/outputs/EPIC/visualizations/new_alg/new_alg_low_conf_start_from_all.avi", fps=10)
        #self.viewer.enable_videowriter("/home/drussel1/dev/deep_sort/outputs/ADL/track_visualizations/new_alg/new_alg_low_conf_start_from_all_unmatched.avi", fps=10)
        #self.viewer.enable_videowriter("/home/drussel1/dev/deep_sort/outputs/ADL/track_visualizations/new_alg/new_alg_no_low_conf_start_from_all_unmatched.avi", fps=10)
        #self.viewer.enable_videowriter("/home/drussel1/dev/deep_sort/outputs/ADL/track_visualizations/new_alg/new_alg_low_conf_start_from_first_unmatched.avi", fps=10)
        #self.viewer.enable_videowriter("/home/drussel1/dev/deep_sort/outputs/ADL/track_visualizations/new_alg/no_new_alg_vis_one_track.avi", fps=10)
        self.frame_idx = seq_info["min_frame_idx"]
        self.last_idx = seq_info["max_frame_idx"]

    def run(self, frame_callback, good_frames):
        """only track the good frames and skip the other ones"""
        # pass the callback and the goodframes to the self.update function
        # I think this lambda is unneccesary
        self.viewer.run(lambda: self._update_fun(frame_callback, good_frames=good_frames))

    # MOD display was set to False
    def _update_fun(self, frame_callback, good_frames=None, display=True):
        print("in _update_fun the value of self.frame_idx is {}".format(self.frame_idx))
        if self.frame_idx > self.last_idx:
            skip_frame = False
            return False, skip_frame  # Terminate
        if good_frames is not None and self.frame_idx not in good_frames:
            self.frame_idx += 1
            skip_frame = True
            return True, skip_frame # simply do nothing on this itteration
        if display:
            frame_callback(self, self.frame_idx)
        self.frame_idx += 1
        skip_frame = False
        return True, skip_frame

    def set_image(self, image):
        self.viewer.image = image

    def draw_groundtruth(self, track_ids, boxes):
        for track_id, box in zip(track_ids, boxes):
            #input((track_id, box))
            #create the wider black boundary
            self.viewer.color = (0.0, 0.0, 0.0) 
            self.viewer.thickness = 8
            self.viewer.rectangle(*box.astype(np.int), label='')
            # and the uniquely-colored interior
            self.viewer.color = create_unique_color_uchar(track_id)
            self.viewer.thickness = 2
            self.viewer.rectangle(*box.astype(np.int), label=str(track_id))

    def draw_detections(self, detections):
        self.viewer.thickness = 2
        self.viewer.color = 0, 0, 255
        for i, detection in enumerate(detections):
            self.viewer.rectangle(*detection.tlwh)

    def draw_trackers(self, tracks):
        ONLY_SHOW_ONE = True
        if ONLY_SHOW_ONE: # I believe these tracks are sorted w.r.t. to seniority, so this should handle it niavely
            # check if the one we want to visualize
            confirmed_ids = [track.track_id for track in tracks if track.is_confirmed]

            if self.index_to_vis not in confirmed_ids: # the track must have died
                self.index_to_vis = random.choice(confirmed_ids)

            track = [t for t in tracks if t.track_id == self.index_to_vis][0] # this is the cleanest way I found to get the item

            self.viewer.color = create_unique_color_uchar(track.track_id)
            if track.time_since_update > 0:
                self.viewer.thickness = 2
            else:
                self.viewer.thickness = 5
            self.viewer.rectangle(
                *track.to_tlwh().astype(np.int), label=str(track.track_id))
  
        else:
            for track in tracks:
                if not track.is_confirmed():# or track.time_since_update > 0:
                    continue
                self.viewer.color = create_unique_color_uchar(track.track_id)
                if track.time_since_update > 0:
                    self.viewer.thickness = 2
                else:
                    self.viewer.thickness = 5
                self.viewer.rectangle(
                    *track.to_tlwh().astype(np.int), label=str(track.track_id))

            #if track.time_since_update > 0:
            #    self.viewer.color = (255, 255, 255)
            #    self.viewer.thickness = 2
            #    self.viewer.rectangle(
            #        *track.to_tlwh().astype(np.int), label=str(track.track_id))
            #self.viewer.gaussian(track.mean[:2], track.covariance[:2, :2],
            #                     label="%d" % track.track_id)
#
