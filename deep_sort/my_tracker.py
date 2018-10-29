# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from . import tools
from . cosine_metric_learning import cosine_inference
from .track import Track
#TODO import the cosine extractor

import cv2


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

    def __init__(self, metric,  max_iou_distance=0.7, max_age=30, n_init=3): #KEY these are really important parameters 
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1
        self._embedder = cosine_inference.CosineInference()
        #self.cosine_embbeder = 

    def predict(self): # this doesn't need to be changed at all
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections, **kwargs): # this is the root of what needs to be changed
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        # here I want to know what the type of the unmatched track variable is 
        #TODO
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # TODO here all we need is the unmatched tracks, ALL the detections in the frame, and the images
        # also we need the cosine extractor and some sort of cropper object
        # actually as a first pass we could just do it based on IOU
        # also this should be done properly with a flag which can turn in on of off
        # use kwargs for in images and the detections
        
        print('matches {}, unmatched_tracks {}, unmatched_detections {}'.format(matches, unmatched_tracks, unmatched_detections))
        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])

        for track_idx in [i for i in unmatched_tracks if self.tracks[i].is_tentative()]: # we arent' going to do a search with these ones, I don't think
            self.tracks[track_idx].mark_missed()

        # yes, this could be a lambda, but this seems more readable for now
        # TODO this should really be a function  
        TRY_TO_RECOVER=True
        if [self.tracks[i] for i in unmatched_tracks if self.tracks[i].is_confirmed()] != [] and TRY_TO_RECOVER: # the set of unmatched confirmed tracks
            print("retrying {}".format([detections[i] for i in unmatched_detections]))

            def get_conf(detection):
                return detection.confidence

            bad_detections = sorted(kwargs["bad_detections"] + [detections[i] for i in unmatched_detections], key=get_conf)
            confirmed_tracks = [
                i for i, t in enumerate(self.tracks) if t.is_confirmed()]
            #pdb.set_trace()
            for bd in bad_detections:
                matches, new_unmatched_tracks, unmatched_detections = \
                            linear_assignment.matching_cascade(
                                    self.gated_metric, self.metric.matching_threshold, self.max_age, 
                                [self.tracks[i] for i in unmatched_tracks if self.tracks[i].is_confirmed()], [bd])#, [i for i in unmatched_tracks if self.tracks[i].is_confirmed()])
                #[i for i in unmatched_tracks if self.tracks[i].is_confirmed()]
                confirmed_tracks = [i for i in unmatched_tracks if self.tracks[i].is_confirmed()] 
                assert len(matches) <= 1, "Since we are looking at only one detection at a time, there shouldn't be more matches, but instead it is {}".format(matches)
                TRACK_IND=0
                DET_IND=1
                FIRST_MATCH=0
                if len(matches) == 1: # only try to update if there's a match, otherwise you'll get a nice out of bounds error
                    self.tracks[unmatched_tracks[matches[FIRST_MATCH][TRACK_IND]]].update(
                        self.kf, bd) # indexing into the list of tracks by the location in the list of indeices passed in
                # at this point the status isn't being updated, thought potentially it should be.
                #update the unmatched tracks
                unmatched_tracks = [confirmed_tracks[i] for i in new_unmatched_tracks]
                if new_unmatched_tracks == []:
                    break


        for track_idx in unmatched_tracks: # these places are where the births and deaths start, but they aren't finalized until later. I need to find that place
            self.tracks[track_idx].mark_missed()
        #for bd in bad_detections:
        #    tlbr_bb = tools.tlwh_to_tlbr(bd.tlwh)
        #    # here we need to extract the descriptors
        #    # which requires a cropping step and utilizing the descriptor thing
        #    # yes it's a bit brutish to just put everything to int, but I doubt it matters
        #    #remember that images are indexed in an i, j convention rather than x first
        #    crop = kwargs["image"][int(tlbr_bb[1]):int(tlbr_bb[3]), int(tlbr_bb[0]):int(tlbr_bb[2])] 
        #    # perhaps unnecessary copy here 
        #    crop = cv2.resize(crop.copy(), (128, 128)) # TODO decide if I need flag here for a non-default style
        #    cv2.destroyAllWindows()
        #    cv2.imshow("crop", crop)
        #    crop = np.expand_dims(crop, 0)
        #    features.append(self._embedder.get_features(crop)[0])

             # we don't want to compute features multiple times
             # this is where I want to match with the low conf detections
             # I'm going to allow mulitiple tracks to match with the same detection
             # You want to compute a list of features, but not all of them at once
             # TODO 
             #this is where we actually do the matching
             # It seems like I can just do the same sort of matching as done before, with all feasible tracks and the new detection
             # It would be computationally better to remove detections which don't overlap, but this will be handled implicitly by the gating function
          

        # get the deleted tracks
        deleted_tracks = [t for t in self.tracks if t.is_deleted()]
        # the filter them out from the list
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        # and now do matching
        if len(deleted_tracks) > 0:
            for track in self.tracks:
                print( 'DEATH ID: {}, state: {}, strong: {}, stack: {}'.format(track.track_id, track.state, track.is_strong(), track.occluded_stack))

        for deleted_track in deleted_tracks:
            # we only want to match with a good track so require_strong is true
            best_occluder = self.get_max_overlap(deleted_track.to_tlbr(), require_strong=True)
            num_adds = 0
            for track in self.tracks:
                if track.track_id == best_occluder:# there is no other way to index them with the simple list
                    track.add_occluded(deleted_track.track_id)
                    assert num_adds == 0 # it shouldn't add the same thing twice
                    num_adds += 1

            #print('the best occluder candidate for {} is {}'.format(deleted_track.track_id, best_occluder))

        # it seems tracks should be initialized after killing the other ones
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])


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

    def gated_metric(self, tracks, dets, track_indices, detection_indices):
        features = np.array([dets[i].feature for i in detection_indices])
        targets = np.array([tracks[i].track_id for i in track_indices])
        cost_matrix = self.metric.distance(features, targets)
        cost_matrix = linear_assignment.gate_cost_matrix(
            self.kf, cost_matrix, tracks, dets, track_indices,
            detection_indices)

        return cost_matrix


    def _match(self, detections):


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
        nearest_occluder = self.get_max_overlap(detection.to_tlbr(), require_occluded=True)
        for track in self.tracks:
            print( ' BIRTH ID: {}, state: {}, strong: {}, stack: {}'.format(track.track_id, track.state, track.is_strong(), track.occluded_stack))
        occluded_id = -1
        for track in self.tracks:
            if track.track_id == nearest_occluder:
                #print('about to remove')
                occluded_id = track.remove_occluded()
                #input('removed')

        #if nearest_occluder != -1:
        #    input('the index of self.tracks[{} -1] is {}'.format(nearest_occluder, self.tracks[nearest_occluder - 1].track_id)) # the indices were 1 indexed
        #print('The nearest occluder to {} is {}'.format(detection.to_tlbr(), nearest_occluder))

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
