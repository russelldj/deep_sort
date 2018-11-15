# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from . import tools
from .track import Track
from .detection import Detection
from .tools import ltwh_to_tlbr, tlbr_to_ltwh
from scipy import stats


import multiprocessing as mp
#TODO import the cosine extractor

import cv2
import pandas as pd
import time
import pdb

class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.
    """
    # TODO
    # determine where tracks die
    # And what the underlying data structure of the track object is
    # first need to add the que 
    ## I'm not sure that it is the correct approach to use a que, but it seems as likely to work as the other ones
    # Then do the death association
    # then the birth

    def __init__(self, metric,  max_iou_distance=0.7, max_age=30, n_init=3, tracker_type="deep_sort", flow_dir=""): #KEY these are really important parameters 
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1
        tracker_names = ["flow_only_tracker", "deep_sort", "flow_tracker", "flow_matcher"];
        if tracker_type not in tracker_names:
            raise ValueError("The tracker type was {} when it should have been one of: {}".format(tracker_type, tracker_names))
        self.flow_tracker_names = tracker_names[1:]
        self.tracker_type = tracker_type
        self.flow_dir = flow_dir
        self.flow = None
        # TODO, test the effects of this
        self.OCCLUDER_STACK = False # use my initial hacky occluder thing
        self.USE_MODE = False


    def predict(self): # this doesn't need to be changed at all
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        # this could do the prediction with the flow pretty easily
        if self.tracker_type == "flow_only_tracker": 
            for track in self.tracks:
                track.flow_predict(self.flow)
            #this isn't quite right because there are two flavors of the flow algorithm, the one where flow is used for prediction and the other where it is just for matching
        elif self.tracker_type == "deep_sort" or self.tracker_type=="flow_tracker":
            for track in self.tracks:
                track.predict(self.kf)

    def update(self, detections, **kwargs): # this is the root of what needs to be changed
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """

        # sketch of the algorithms decision tree 
        # use the cannonical deep sort matching thing to generate matches, unmatched tracks, and unmatched detections
        # if using my modification, match some more tracks to detections, potentially low-conf ones
        # if propogatign by flow, do that for all tracks that you don't want to kill

        #TODO try to fix the organization of this so it's a bit less heinous
        # TODO a lot of that could be accomplished by putting some stuff in functions

        self.load_frame_flow(kwargs["frame_idx"])

        #Do the matching either with flow or appearance
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)
        print('matches {}, unmatched_tracks {}, unmatched_detections {}'.format(matches, unmatched_tracks, unmatched_detections))
        
        # Update track set with the first round of matches
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])

        #TODO this should really be changed to calling the flow_predict method for each track, and be moved later in the program
        # predict all of the unmatched ones with the flow, if they aren't super new

        # this should also get moved later
       
        USE_LOW_CONF = True
        assert kwargs["use_unmatched"], "for now, just assume this is thei default"
        if kwargs["use_unmatched"]:
            #unmatched_tracks is really the indices
            if USE_LOW_CONF:
                unmatched_track = self.retry_detections(unmatched_tracks, [detections[i] for i in unmatched_detections], kwargs["bad_detections"], initialize_new_tracks=True)
            else:
                unmatched_track = self.retry_detections(unmatched_tracks, [detections[i] for i in unmatched_detections], [], initialize_new_tracks=True)
        else:
            #TODO loop thorugh all of the bad detections
            for ud in [detections[i] for i in unmatched_detections]:
                self._initiate_track(ud)

        for unmatched_track in unmatched_tracks: # these places are where the births and deaths start, but they aren't finalized until later. I need to find that place
            
            #assert not self.tracks[unmatched_track].is_tentative()
            if unmatched_track == 205:
                import pdb; pdb.set_trace()
            if self.tracker_type == "flow_tracker": 
                # move the track based on the flow
                self.flow_VOT(self.tracks[unmatched_track])
            # this ordering is important, you don't want to mark missed first
            self.tracks[unmatched_track].mark_missed()
        
        # get the deleted tracks
        deleted_tracks = [t for t in self.tracks if t.is_deleted()]
        # the filter them out from the list
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        # and now do matching
        if len(deleted_tracks) > 0:
            for track in self.tracks:
                print( 'DEATH ID: {}, state: {}, strong: {}, stack: {}'.format(track.track_id, track.state, track.is_strong(), track.occluded_stack))
        
        if self.OCCLUDER_STACK:
            for deleted_track in deleted_tracks:
                # we only want to match with a good track so require_strong is true
                best_occluder = self.get_max_overlap(deleted_track.to_tlbr(), require_strong=True)
                num_adds = 0
                for track in self.tracks:
                    if track.track_id == best_occluder:# there is no other way to index them with the simple list
                        track.add_occluded(deleted_track.track_id)
                        assert num_adds == 0 # it shouldn't add the same thing twice
                        num_adds += 1

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)
        #this is the end of update

    
    def flow_VOT(self, track, scale_factor=1):
        """
        predicts the movement of a track from the optical flow
        params
        ----------
        track : deep_sort.Track
            This is the track that needs to updated. Likely it will come from the unmatched but confirmed set
        return
        ---------- 
        None
            The state of the track will be updated
        #>>> tracker = Tracker("metric")
        """
        assert self.flow is not None, "the flow should have been loaded in update"
        def fint(float_, axis_num=0):
            fint = int(np.floor(float_))
            return(np.clip(fint, 0, self.flow.shape[axis_num]))

        assert self.flow.shape[2] == 2, "the flow should be x, y displacement vectors"

        bbox = ltwh_to_tlbr(track.to_tlwh()) #to_tlwh is a misnomer
        left   = min(fint(bbox[1]), fint(bbox[3]))  # they can be inverted by the filter
        top    = min(fint(bbox[0]), fint(bbox[2]))  # they can be inverted by the filter
        right  = max(fint(bbox[1]), fint(bbox[3]))  # they can be inverted by the filter
        bottom = max(fint(bbox[0]), fint(bbox[2]))  # they can be inverted by the filter

        tracked_region = self.flow[top:bottom, left:right]
        y_size, x_size, _ = tracked_region.shape
        if x_size == 0 or y_size == 0:
            print("The track had zero width or was out or frame. it is in ltrb {}".format([left, top, bottom, right]))
        else:
            if self.USE_MODE:
                x_ave = stats.mode(tracked_region[...,0].astype(int), axis=None).mode[0] * scale_factor
                y_ave = stats.mode(tracked_region[...,1].astype(int), axis=None).mode[0] * scale_factor
            else:
                x_ave = np.average(tracked_region[...,0]) * scale_factor
                y_ave = np.average(tracked_region[...,1]) * scale_factor

            self.x_hist = np.histogram(tracked_region[...,0])
            self.y_hist = np.histogram(tracked_region[...,1])

            #x_ceoffs = np.polyfit(np.arange(x_size), tracked_region[int(y_size / 2), :, 0], 1)
            #y_ceoffs = np.polyfit(np.arange(y_size), tracked_region[:, int(x_size/ 2), 1], 1)

            #warning this might be weird for python 2
            left   += x_ave #- x_ceoffs[0] * x_size / 2.0 
            right  += x_ave #+ x_ceoffs[0] * x_size / 2.0
            top    += y_ave #- y_ceoffs[0] * y_size / 2.0
            bottom += y_ave #+ y_ceoffs[0] * y_size / 2.0
        # this whole switching thing might be hard with the filter
        tlwh_bbox = tlbr_to_ltwh([top, left, bottom, right])
        # there should be a set for cropping and another for maintaining accuracy
        old_tlwh_bbox = tlwh_bbox.copy()
        track.flow_update(self.kf, tlwh_bbox) # this is an issue, I probably need to write another method, because it doesn't make sense to create a detection with some null feature
        assert track.location.tolist() == old_tlwh_bbox.tolist()


    def retry_detections(self, unmatched_track_idxs_, initial_unmatched_detections_, otherwise_excluded_detections_, initialize_new_tracks=False): # It makes sense to do it like this because these are the detections which are most likely to be useful, and we shouldn't mix in the subpar ones yet
        # all detections that get passed in should be used 
        # all tracks which are confirmed should be used
        # at the end it needs to return matches, unmatched tracks, and unmatched detections
        # likely this will be in the form of actual tracks and detections, to avoid confusion and because everything is just a reference anyway
        def get_conf(detection):
            return detection.confidence
        
        # do some hacking to deal with the two sorts of detections
        unmatched_detections_ = [] 
        for unmatched_detection in initial_unmatched_detections_:
            unmatched_detection.was_NMS_suppressed = False # this is only for the bad ones suppressed by NMS or confidence
            unmatched_detections_.append(unmatched_detection)

        for suppressed_detection in otherwise_excluded_detections_:
            suppressed_detection.was_NMS_suppressed = True # this is the other alternative
            unmatched_detections_.append(suppressed_detection)
            
        detections_ = sorted(unmatched_detections_, key=get_conf)

        confirmed_unmatched_tracks_ = [self.tracks[i] for i in unmatched_track_idxs_ if self.tracks[i].is_confirmed()]
        confirmed_unmatched_track_inxs_ = [i for i in unmatched_track_idxs_ if self.tracks[i].is_confirmed()]
        unmatched_track_idxs_ = None # make sure this doesn't get used naively agian

        final_unmatched_detections_ = []

        for ud in unmatched_detections_:
            # the detection here isn't really useful
            matches, new_unmatched_tracks, new_unmatched_detections = \
                        linear_assignment.matching_cascade(
                                self.gated_metric, self.metric.matching_threshold, self.max_age, 
                                confirmed_unmatched_tracks_, [ud]) # this can't be easily cached as unmatched tracks keeps getting shorter
            assert len(matches) <= 1, "there is only one detection, so there shouldn't be more than one match"
            if len(matches) == 1:
                TRACK_LOCATION = 0
                ONLY_MATCH = 0
                confirmed_unmatched_tracks_[matches[ONLY_MATCH][TRACK_LOCATION]].update(self.kf, ud) # update the relavanent track
                # HACK
            elif initialize_new_tracks:
                #mark that this detection was missed
                final_unmatched_detections_.append(ud)
                
            confirmed_unmatched_tracks_ = [confirmed_unmatched_tracks_[i] for i, _ in enumerate(confirmed_unmatched_tracks_) if i not in new_unmatched_tracks] # this array will get returned
            BREAK_EARLY = False # it appears that this is a bad idea
            assert not BREAK_EARLY, "just for now, I don't want this on"
            if BREAK_EARLY:
                if len(confirmed_unmatched_tracks_) == 0:
                    break # we are done, though maybe all of the detections should actually be matched confirmed_unmatched_track_inxs_ = [self.tracks.index(c_u_t) for c_u_t in confirmed_unmatched_tracks_]
        non_NMS_final_unmatched_detections = [ud for ud in final_unmatched_detections_ if not ud.was_NMS_suppressed]
        for ud in non_NMS_final_unmatched_detections:
            self._initiate_track(ud)

        #input("non_NMS_final_unmatched_detections {}\n, initial_unmatched_detections_ {}\n bad_detections {}".format(non_NMS_final_unmatched_detections, initial_unmatched_detections_,otherwise_excluded_detections_))
        return confirmed_unmatched_track_inxs_

    def gated_metric(self, tracks, dets, track_indices, detection_indices):
        features = np.array([dets[i].feature for i in detection_indices])
        targets = np.array([tracks[i].track_id for i in track_indices])
        cost_matrix = self.metric.distance(features, targets)
        cost_matrix = linear_assignment.gate_cost_matrix(
            self.kf, cost_matrix, tracks, dets, track_indices,
            detection_indices)

        return cost_matrix


    def flow_metric(self, tracks, dets, track_indices, detection_indices):
        dets = np.array([dets[i].tlwh for i in detection_indices])
        targets = np.array([tracks[i].to_tlwh() for i in track_indices])
        #use the flow which is an instance variable, and the locations of the dets and tracks
        cost = self.compute_cost(self.flow, targets, dets)
        return cost


    def _match(self, detections):
        #pdb.set_trace()
        if self.tracker_type == "flow_matcher":
            matches, unmatched_tracks, unmatched_detections = \
                linear_assignment.min_cost_matching(
                    self.flow_metric, self.max_iou_distance,
                    self.tracks, detections)

        else:  # use the normal appearance features approach
            # Split track set into confirmed and unconfirmed tracks.
            confirmed_tracks = [
                i for i, t in enumerate(self.tracks) if t.is_confirmed()]
            unconfirmed_tracks = [
                i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

            # Associate confirmed tracks using appearance features.
            matches_a, unmatched_tracks_a, unmatched_detections = \
                linear_assignment.matching_cascade(
                    self.gated_metric, self.metric.matching_threshold, self.max_age,
                    self.tracks, detections, confirmed_tracks)

            # Associate remaining tracks together with unconfirmed tracks using IOU.
            iou_track_candidates = unconfirmed_tracks + [
                k for k in unmatched_tracks_a if
                self.tracks[k].time_since_update == 1]
            unmatched_tracks_a = [
                k for k in unmatched_tracks_a if
                self.tracks[k].time_since_update != 1]
            matches_b, unmatched_tracks_b, unmatched_detections = \
                linear_assignment.min_cost_matching(
                    iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                    detections, iou_track_candidates, unmatched_detections)

            matches = matches_a + matches_b
            unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        # this is where to search for nearby tracks which are occluding something
        occluded_id = -1

        if self.OCCLUDER_STACK: # Potentially find the ID from a stack of occluded ones
            nearest_occluder = self.get_max_overlap(detection.to_tlbr(), require_occluded=True)
            for track in self.tracks:
                if track.track_id == nearest_occluder:
                    occluded_id = track.remove_occluded()

        # this should be tracker function which takes the self.tracks list and the new birthed one
        mean, covariance = self.kf.initiate(detection.to_xyah())
        if occluded_id != -1:
            self.tracks.append(Track(
                mean, covariance, occluded_id, self.n_init, self.max_age,
                detection.feature))
        else: 
            self.tracks.append(Track(
                mean, covariance, self._next_id, self.n_init, self.max_age,
                detection.feature))
            self._next_id += 1

    def get_max_overlap(self, occluded_box, require_strong=False, require_occluded=False, distance_threshold=1):
        """ this could be used for other stuff, like births
        input:
        occluded_box the bounding box which either died or was detected in the tlbr format
        require_strong: only match with confirmed and recently detected boxes
        require_occluded: only used for births; match with the nearest object which is occluding another one
        
        """
        #TODO make a better method for differentiating between two tracks which both fully overlap
        def compute_overlap(target_box, background_box):
            """This function computes how much of the target box is covered by the background one. Inherently oriented. Input should be in the form tlbr"""
            x_overlap = max(0, min(target_box[3], background_box[3]) - max(target_box[1], background_box[1]))
            y_overlap = max(0, min(target_box[2], background_box[2]) - max(target_box[0], background_box[0]))
            overlap_area = x_overlap * y_overlap
            target_area = (target_box[2] - target_box[0]) * (target_box[3] - target_box[1])
            return overlap_area / float(target_area)
        highest_overlap = 0.0
        best_track = -1
        # Important
        for track in self.tracks:
            if not track.is_deleted():# TODO and (track.track_id == ):
                current_overlap = compute_overlap(occluded_box, track.to_tlbr())
                # just don't think too much about this line and hope that it
                # real talk, the two cases are that we required there to be an occluded object and there wasn't one, this is for birth
                # alternatively, we wanted a track we were confident about and there wasn't one
                if (require_occluded and not track.has_occluded()) or (require_strong and not track.is_strong()):
                    print('about to skip one which was not occluding or was not strong')
                    continue # jump to the next itteration of the loop and skip the update
                if current_overlap > highest_overlap:
                    best_track = track.track_id
                    highest_overlap = current_overlap
        return best_track

# Flow related section

    def compute_cost(self, flow, track_boxes, det_boxes):
        """
        params
        flow : np.array
            This is the M x N x 2 flow representation, where the first channel is x and the second is y
        track_boxes : List[]
            This should be the filter-predicted location in the format xywh
        det_boxes : List[]
            All the thresholded detections
        returns
        cost : np.array
            This is the pairwise cost function with tracks on the i axis and detections on the j
        #>>> flow = np.zeros((480, 640)) 
        #... track_boxes = [[0,0,100,100], [0, 100, 100, 100], [100, 0, 100, 100]]
        #... det_boxes   = [[200, 0, 100, 100], [0, 0, 100, 100]]
        #... compute_cost(flow, track_boxes, det_boxes)
        #None
        """
        cost = np.zeros((len(det_boxes), 0), dtype=int)
        det_sizes = [d[2] * d[3] for d in det_boxes]
        #import pdb; pdb.set_trace()
        def ifint(float_):
            # the kalman filter can predict that the object will move out of frame, so the values need to be clamped to the size the flow (and the iamge)
            return int(np.clip(np.floor(float_), 0, flow.shape[0]-1))

        def jfint(float_):
            # the kalman filter can predict that the object will move out of frame, so the values need to be clamped to the size the flow (and the iamge)
            return int(np.clip(np.floor(float_), 0, flow.shape[1]-1))

        for track in track_boxes:
            det_offsets = self.compute_offsets(track, det_boxes)
            num_in_each = np.zeros((len(det_offsets)), dtype=int)
            max_x_flow = -10000
            for i_idx in range(ifint(track[3])):
                for j_idx in range(jfint(track[2])):
                    flow_pixel = flow[i_idx, j_idx, :]
                    max_x_flow = max(max_x_flow, flow_pixel[0])
                    #print('flow pixel {}'.format(flow_pixel))
                    # tally which detection each flow pixel lands in
                    num_in_each += self.check_offsets(i_idx, j_idx, flow_pixel, det_offsets)
                    
            # there needs to some sort of normalized w.r.t. to the area of the dections and the tracks
            #TODO normalize the values of num_in_each 
            track_size = track[2] * track[3]
            # make this metric as similar to the  standard IOU metric as possible
            # like IOU, this metric should be bounded by [0,1]
            normalization_factor = np.asarray([det_size + track_size - num_in_each[inx] for inx, det_size in enumerate(det_sizes)])
            num_in_each = np.divide(num_in_each, normalization_factor)
            cost = np.append(cost, np.expand_dims(num_in_each, axis=1), axis=1)

        cost = np.transpose(cost) # IMPORTANT make sure the i axis is the length of the track vector
        return 1 - cost # this is because higher IOU is better but the matching is posed as a cost
    #def compute_cost(self, flow, track_boxes, det_boxes):
    #    """
    #    params
    #    flow : np.array
    #        This is the M x N x 2 flow representation, where the first channel is x and the second is y
    #    track_boxes : List[]
    #        This should be the filter-predicted location in the format xywh
    #    det_boxes : List[]
    #        All the thresholded detections
    #    returns
    #    cost : np.array
    #        This is the pairwise cost function with tracks on the i axis and detections on the j
    #    #>>> flow = np.zeros((480, 640)) 
    #    #... track_boxes = [[0,0,100,100], [0, 100, 100, 100], [100, 0, 100, 100]]
    #    #... det_boxes   = [[200, 0, 100, 100], [0, 0, 100, 100]]
    #    #... compute_cost(flow, track_boxes, det_boxes)
    #    #None
    #    """
    #    start = time.time()
    #    cost = np.zeros((len(det_boxes), 0), dtype=int)
    #    det_sizes = [d[2] * d[3] for d in det_boxes]
    #    #import pdb; pdb.set_trace()
    #    def ifint(float_):
    #        # the kalman filter can predict that the object will move out of frame, so the values need to be clamped to the size the flow (and the iamge)
    #        return int(np.clip(np.floor(float_), 0, flow.shape[0]-1))

    #    def jfint(float_):
    #        # the kalman filter can predict that the object will move out of frame, so the values need to be clamped to the size the flow (and the iamge)
    #        return int(np.clip(np.floor(float_), 0, flow.shape[1]-1))
    #     
    #    def compute_overlaps(det_offsets, flow, track, i_idx, output=None):
    #        num_in_each = np.zeros((len(det_offsets)), dtype=int)
    #        #row_start = time.time()
    #        for j_idx in range(jfint(track[2])):
    #            flow_pixel = flow[i_idx, j_idx, :]
    #            #print('flow pixel {}'.format(flow_pixel))
    #            # tally which detection each flow pixel lands in
    #            #TODO check if this isn't efficient
    #            num_in_each += self.check_offsets(i_idx, j_idx, flow_pixel, det_offsets)
    #        if output is not None:
    #            output.put(num_in_each)
    #        else:
    #            return num_in_each
    #        #print("row took {} seconds".format(time.time() - row_start))
    #    
    #    output = []
    #    for track in track_boxes:
    #        for i_idx in range(ifint(track[3])):
    #            det_offsets = self.compute_offsets(track, det_boxes)
    #            output.append(compute_overlaps(det_offsets, flow, track, i_idx))


    #            #output = mp.Queue()

    #            #processes = [mp.Process(target=compute_overlaps, args=(det_offsets, flow, track, i_idx, output)) for i_idx in range(ifint(track[3]))]
    #            #
    #            #output_start = time.time()
    #            ## Run processes
    #            #for p in processes:
    #            #    p.start()

    #            ## Exit the completed processes
    #            #for p in processes:
    #            #    p.join()
    #            #output = [output.get() for p in processes]
    #            #print("computing the output took {} seconds".format(time.time() - output_start))
    #            num_in_each = sum(output)
    #            # there needs to some sort of normalized w.r.t. to the area of the dections and the tracks
    #            #TODO normalize the values of num_in_each 
    #            track_size = track[2] * track[3]
    #            # make this metric as similar to the  standard IOU metric as possible
    #            # like IOU, this metric should be bounded by [0,1]

    #            normalization_factor = np.asarray([det_size + track_size - num_in_each[inx] for inx, det_size in enumerate(det_sizes)])
    #            num_in_each = np.divide(num_in_each, normalization_factor)
    #            cost = np.append(cost, np.expand_dims(num_in_each, axis=1), axis=1)

    #    cost = np.transpose(cost) # IMPORTANT make sure the i axis is the length of the track vector
    #    print("computing the cost took {} seconds".format(time.time() - start))
    #    return 1 - cost # this is because higher IOU is better but the matching is posed as a cost


    def check_offsets(self, i_idx, j_idx, flow_pixel, det_offsets):
        """
        params:
                                ---------- 
        i_idx : int
            location in the track
        j_idx : int 
            location in the track
        flow_pixel : ArrayLike
            the x and y component of the flow
        offsets : List[ArrayLike]
            the offsets between the top left corner of the track and the detections

        returns
        ----------
        matches : List[Bool]

        """
        #check that this is correct, the y might be first
        x_offset = j_idx + flow_pixel[0]
        y_offset = i_idx + flow_pixel[1]
        #matches = []
        #for det_offset in det_offsets:
        #    if x_offset >= det_offset[0] and x_offset <= det_offset[2] and \
        #        y_offset >= det_offset[1] and y_offset <= det_offset[3]:
        #        matches.append(True)
        #    else:
        #        #print("x_offset: {}, y_offset: {}".format(x_offset, y_offset))
        #        matches.append(False)

        matches = [True if x_offset >= det_offset[0] and x_offset <= det_offset[2] and y_offset >= det_offset[1] and y_offset <= det_offset[3] else False for det_offset in det_offsets]
        return np.array(matches, dtype=int)

    def compute_offsets(self, track_box, det_boxes):
        """
        params:
                                ----------  
        track_box : ArrayLike
            This should be in the form [l, t, w, h]
        det_boxes : List[ArrayLike] 
            This is a list of all detections in the same form as above
        returns
        ----------  
        offsets : List[ArrayLike] 
            The offset between the top left corner of the track and the top left and bottom right corners of the detection.

        """
        offsets = []
        for det_box in det_boxes:
            offset = (det_box[0] - track_box[0],\
            det_box[1] - track_box[1],\
            det_box[0] + det_box[2] - track_box[0],\
            det_box[1] + det_box[3] - track_box[1])
            offsets.append(offset)
        return offsets
   

    def load_frame_flow(self, frame_idx):
        self.frame_idx = frame_idx
        flow_prefix = "{}/{:06d}".format(self.flow_dir, self.frame_idx)
        self.flow = self.load_flow(flow_prefix)


    def load_flow(self, flow_prefix):
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
         
        #WARNING, this was inverted
        return np.concatenate((-1 * x_flow[...,0:1], -1 * y_flow[...,0:1]), axis=2) # keep the dimensionality with 0:1

