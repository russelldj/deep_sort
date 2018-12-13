# vim: expandtab:ts=4:sw=4
from .tools import ltwh_to_xyah, xyah_to_ltwh, ltwh_to_tlbr
from . import mask as MaskTools
from .DaSiamRPN.code.SiamRPN_tracker import SiamRPN_tracker
import numpy as np
from scipy import stats
import logging
import cv2
import pdb

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.
    `occluded` might be used sometime if I want to mark tracks behind other ones 

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3
    Occluded = 4

class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.
    occluded_stack : Queue[int]
        A FILO stack which holds the list of occluded object instances.
        These represent the objects which are thought to be behind the current object

    """

    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None, image=None, is_flow_track=False):
        # TODO make sure all of the args are being parsed correctly
        if image is not None:
            assert image.shape[2] == 3
        self.mean = mean # this is the location if it is a flow track in the format [l,t,r,b]
        self.location = None #ltwh_to_tlbr(xyah_[:4]))
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.is_flow_track = is_flow_track # this means that the kalman filter won't be used

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self.tracker = cv2.TrackerMOSSE_create()

        # TODO have a method which returns if the track is being moved by the tracker
        #TODO this needs to be replaced with initializing the DaSiam tracker

        self.occluded_stack = []

        self._n_init = n_init
        self._max_age = max_age
        self.use_location = False
        self.mask = MaskTools.bbox_to_contour(self.to_tlwh())
        self.init_tracker(image)

    def init_tracker(self, image):
        # reset the tracker location with the new image and the current tracker state
        #TODO also the tracker initialization
        #TODO determine why this is getting initialize with feature some times
        assert len(image.shape) == 3
        self.Siam_tracker = SiamRPN_tracker(image, tuple(self.to_tlwh().tolist()))
        self.tracker.init(image, tuple(self.to_tlwh().tolist()))
   
    def tracker_update(self, image):
        pdb.set_trace()
        ok, bbox = self.tracker_predict(image)
        if ok: # there wasn't a tracking 
            print("ok")

    def tracker_predict(self, image):
        #split this into two parts, the updating and the grabbing of the information
        score, bbox = self.Siam_tracker.predict(image)
        ok = score > 0.8 #HACK this is just a temporary thing to see what happens if you keep tracking

        #the results from the earlier results will be overwritten
        #ok, bbox = self.tracker.update(image)
        return ok, ltwh_to_tlbr(bbox)

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.
        """
        if self.use_location:
            assert self.location is not None, "time is {}".format(self.time_since_update)
            return self.location.copy() # which had better be stored in the ltwh.copy()
        else:
            ret = self.mean[:4].copy()
            ret[2] *= ret[3]
            ret[:2] -= ret[2:] / 2
            return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def flow_predict(self, flow):
        """propogate the location based on the flow and the past bounding box

        parameters
        ----------
        flow : np.array
            mxnx2, with the first channel representing the x displacement and the second y
        self.mean : arraylike
            this should be in the form [l, t, r, b]
        """

        def fint(float_):
            return int(np.floor(float_))

        assert flow.shape[2] == 2, "the flow should be x, y displacement vectors"
        raise NotImplementedError("this isn't completed")
        self.box_predict(flow, self.mean)

    def box_predict(self, flow, bbox, scale_factor=1.0):
        def fint(float_):
            return int(np.floor(float_))

        tracked_region = flow[fint(bbox[1]):fint(bbox[3]),fint(bbox[0]):fint(bbox[2])]
        x_size = fint(bbox[2]) - fint(bbox[0])
        y_size = fint(bbox[3]) - fint(bbox[1])


        USE_MODE = False

        if USE_MODE:
            x_ave = stats.mode(tracked_region[...,0].astype(int), axis=None).mode[0] * scale_factor
            y_ave = stats.mode(tracked_region[...,1].astype(int), axis=None).mode[0] * scale_factor
        else:
             x_ave = np.average(tracked_region[...,0]) * scale_factor
             y_ave = np.average(tracked_region[...,1]) * scale_factor

        x_hist = np.histogram(tracked_region[...,0])
        y_hist = np.histogram(tracked_region[...,1])

        #todo move this to its own method
        #try:
        #    x_ceoffs = np.polyfit(np.arange(x_size), tracked_region[int(y_size / 2), :, 0], 1)
        #    y_ceoffs = np.polyfit(np.arange(y_size), tracked_region[:, int(x_size/ 2), 1], 1)
        #except ValueError:
        #    #determine something more inteligent to do here
        #    input("The box went out of frame")
        logging.debug("x_ave: {}, y_ave: {}".format(x_ave, y_ave))
        #print("xslope: {}, yslope: {}".format(x_ceoffs[0] * x_size, y_ceoffs[0] * y_size))

        #warning this might be weird for python 2
        bbox[0] += x_ave #- x_ceoffs[0] * x_size / 2.0 
        bbox[2] += x_ave #+ x_ceoffs[0] * x_size / 2.0
        bbox[1] += y_ave #- y_ceoffs[0] * y_size / 2.0
        bbox[3] += y_ave #+ y_ceoffs[0] * y_size / 2.0
        return bbox


    def masked_flow(self, flow, polygon):
        print(polygon)
        mask = np.zeros_like(flow[...,0], dtype=np.uint8)
        cv2.fillPoly(mask, polygon.astype(np.int32), 1)
        # now its time to multiply the mask with the x and y flows and then devide my the number in the mask
        num_masked_points = float(sum(sum(mask)))  # sum is applied column(I think)-wise

        return (np.multiply(mask, flow[...,0]), np.multiply(mask, flow[...,1]) , num_masked_points)


    def predict(self, kf):
        """propagate the state distribution to the current time step using a
        kalman filter prediction step.

        parameters
        ----------
        kf : kalman_filter.kalmanfilter
            the kalman filter.

        """
        #todo likewise, this needs to be updated so there's a version where you simply pass in the flow
        #and the internal state is directly updataed
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1


    def flow_update(self, kf, ltwh_bbox, image, feature=None, update_kf=True, update_hit=False):
        """The logic here is evolving but the current approach is as follows:
        If the current feature looks similar, i.e. it is less than max_cosine_distance from a previous feature, perform a REAL update
        * this means that the time_since_update will be set to zero
        Else perform a direct update
        * this will allow the probability mass to spread out and allow detections with other objects
        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        ltwh_bbox : ArrayLike
            The predicted location in the form [left, top, width, height]
        feature : ArrayLike
            This should be a (1, 128) array which represents an embeding of the current flow-predicted location
        kf_update : Bool
            this should be true if the new feature looked similar to the gallery
        """
        # determine how to query the min distance between a feature and the gallery
        # check that value versus the threshold
        # if it is less than that, perform a KF update and set the time since matched to zero
        ## I.E. just call the normal update function
        # else, perform a direct location update and 
        #TODO add the feature if we care about it
        assert len(image.shape) == 3 

        #assert self.is_flow_track, "This isn't set so there will be an issue with to_ltwh"
        logging.warning("doing a flow update")
        # the mask should stay the same here, or be shifted by the flow
        if update_kf:
            self.mean, self.covariance = kf.update(
                self.mean, self.covariance, ltwh_to_xyah(ltwh_bbox)) #TODO make sure this is the same effoect as self.update
            if feature is not None: # this should be sufficiently general, I think this isn't all that important whether there's a new feature
                assert feature[0].shape == (128,) # features come in shaped like (128,1) and we need them to be (128,)
                logging.warning("no longer adding features on flow updates")
                #self.features.append(feature[0])
            if update_hit: # if this is set to True, it effectively waits util it leaves the scene
                self.time_since_update = 0 # now tracks will never die
                self.hits += 1

            if self.state == TrackState.Tentative and self.hits >= self._n_init:
                self.state = TrackState.Confirmed
            self.use_location = False
        else:
            self.location = ltwh_bbox.copy()
            self.use_location = True

        assert image is not None

    def update(self, kf, detection, image):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection, "time is {}".format(self.time_since_update)
            The associated detection.

        """
        #TODO, make a version of detection which doesn't have the filter and is just a direct update
        # OK, I guess what I meant is make the flow_update function better
        # eventually this will be used with masks so just keep that in mind
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        #assert detection.feature.shape == (128,)
        self.features.append(detection.feature)
        self.mask = detection.mask

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed
        self.use_location = False

        #new functionality, update the tracker
        #TODO add the image
        assert image is not None
        self.init_tracker(image)

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
            return True # a track was deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted 
            return True 
        return False # this track was good
    
    def set_confirmed(self):
        """sets the track's status to confirmed
        """
        self.state = TrackState.Confirmed

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted

    def is_strong(self):
        """Return if this track would be visualized"""
        # this line is messy as I copied it from deep_sort_app and modified it so it wasn't is_weak
        return not ((not self.is_confirmed()) or (self.time_since_update > 1))

    def add_occluded(self, occluded_id):
        self.occluded_stack.append(occluded_id)
        print('occluded object {} was added to track {}'.format(occluded_id, self.track_id))

    def remove_occluded(self):
        # BUG this should never be empty
        print(len(self.occluded_stack))
        if self.occluded_stack != []:
            removed_id = self.occluded_stack.pop()
        else:
            print('there was an error where the stack was empty')
            removed_id = -1
        print('occluded object {} was removed from track {}'.format(removed_id, self.track_id))
        print('about to return remove occluded')
        return removed_id

    def has_occluded(self):
        return self.occluded_stack != []
