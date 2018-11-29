from __future__ import division, print_function, absolute_import
import argparse
import os

import cv2
import numpy as np
import os
import pdb

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort.my_tracker import Tracker as MyTracker # make sure to avoid the namespace collision here;
from deep_sort.flow_tracker import FlowTracker
from deep_sort.scorer import Scorer
from deep_sort.tools import ltwh_to_tlbr, tlbr_to_ltwh, ltrb_to_tlbr, safe_crop_ltbr
import pandas as pd
import pdb
import re
import logging
import h5py
import json

def gather_sequence_info(sequence_dir, detection_file, track_class=None, detection_masks=True):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.
    detection_masks : Bool
        Whether the detections have the masks written out

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format. OR a h5 file view if use masks
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    print("\n\n\nTHIS IS A HACK, THE DETECTION FILE IS HARDCODED\n\n\n")
    detection_file = "/home/drussel1/data/ADL/new_mask_outputs/dataset_per_frame/P_18.MP4.h5"
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")
    
    if detection_masks:
        detections = load_masks(detection_file) 
    else:
        detections = load_bboxs(detection_file) 

    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=' ')
    else:
        groundtruth = None

    if len(image_filenames) > 0:
        #input(next(iter(image_filenames.values())))
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None
       
    if detection_masks:
        feature_dim = 0 #this is a HACK as I'm not going to use features with this yet
    else:
        feature_dim = detections.shape[1] - 10 if detections is not None else 0
    
    #TODO build the detections output from the json and have the h5 file returned seperately
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info

def retrieve_frame(h5_file, frame_id):
    frame_id = str(frame_id)
    frame_data = json.loads(h5_file[frame_id].value)
    assert type(frame_data) == dict
    return frame_data

def load_masks(detection_file):
    """
    params
    ---------- 
    detection_file : str
        path to the h5 file which has a dataset for each frame
        The data is encoded with json.dumps so it can be decoded with json.loads

    returns
    ---------- 
    detections : I'm not sure
        the easiest would be to have this just be a h5 file but I'm not sure that is the best
    """
    mask_file = h5py.File(detection_file, 'r')
    return mask_file 

def load_bboxs(detection_file):
    """
    params
    ----------
    detection_file : str
        The path to a detection file containing rectangular bounding boxes in the MOT format and potentially descriptors

    returns
    ---------- 
    detections: np.ndarray
         each row is in the MOT format (I think)
    """
    detections = None
    if detection_file is not None:
        #MOD
        if os.path.isfile(detection_file):
            _, extension = os.path.splitext(detection_file)
            if extension == ".npy":
                detections = np.load(detection_file)
            elif extension == ".h5":
                import h5py
                import pdb; pdb.set_trace()
                detection = h5py.File(detection_file)
            else:
                raise ValueError("The detection file should be .npy or .h5")

            #MOD
            if track_class is not None:
                # only retain the tracks with thei
                inds = detections[:,1] == track_class
                detections = detections[inds, :] # get only the detections which have the target class, which is the second column
                print("Only using tracks from class {}".format(track_class))



        else:
            detections = None
        return detections

def create_groundtruth(groundtruth_mat, frame_idx, min_height=0):
    """Create groundtruths for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of groundtruths. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    (List(np.array), List(np.ndarry)]
        Returns boxes and idices for a given frame index.

    """
    #def ltrb_to_tlwh(box):
        # input x1y1x2y2
        # output
        # return np.array([box[], box[1], box[0] + box[2], box[1] + box[3]])

    frame_indices = groundtruth_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx
        
    box_list = []
    ind_list = []

    for row in groundtruth_mat[mask]:
        box, ind = row[2:6], row[1] 
        if box[3] < min_height:
            continue
        box_list.append(2*box)
        ind_list.append(ind)
    return ind_list, box_list


def create_mask_detections(detection_h5_file, frame_idx, min_height=0, track_class=None, subset_frames=None):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        A h5 file where each dataset is indexed by the frame and can be decoded with json.loads
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.
    subset_frames : Optional[np.array()]
        Only these frames are used

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_dict = retrieve_frame(detection_h5_file, frame_idx)

    detection_list = []
    for index, clas in enumerate(frame_dict["classes"]):
        if track_class is not None and clas != track_class:
            continue #only add the detections of the targe class

        #these boxes are in the form ltrb
        bbox = tlbr_to_ltwh(ltrb_to_tlbr(frame_dict["boxes"][index][:4])) # TODO make sure this is right
        # Turns out it wasn't right, the boxes 
        conf = frame_dict["boxes"][index][4]
        mask = frame_dict["contours"][index] 
        # this is a HACK wherin we create a fake feature with the assumption that it will never be used
        feature = np.zeros((128,))
        feature[0] = 1
        if bbox[3] < min_height:
            continue
        # TODO add another field for detections so they can carry around a nice cute little mask
        detection_list.append(Detection(bbox, conf, feature, mask=mask))
    return detection_list


def create_detections(detection_mat, frame_idx, min_height=0, subset_frames=None):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.
    subset_frames : Optional[np.array()]
        Only these frames are used

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list


def run(sequence_dir, detection_file, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display, stock=False, track_class=None, **kwargs):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.

    """
    if track_class is not None: # this doesn't do anything
        seq_info = gather_sequence_info(sequence_dir, detection_file, track_class)
    else:
        seq_info = gather_sequence_info(sequence_dir, detection_file)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    if stock:
        print('initializing a stock tracker')
        tracker = Tracker(metric, max_age=kwargs['max_age'], max_iou_distance=1.0 - kwargs['min_iou_overlap'])
    else:
        print('initializing a modified tracker')
        # the tracker now has the class as an optional argument
        tracker = MyTracker(metric, max_age=kwargs['max_age'], max_iou_distance=1.0 - kwargs['min_iou_overlap'], tracker_type=kwargs["tracker_type"], flow_dir=kwargs["flow_dir"], update_kf=kwargs["update_kf"], update_hit=kwargs["update_hit"]) # the IOU is inverted as 1 - IOU in the cost matrix

    results = []
    if kwargs["track_subset_file"] is not None:
        good_frames = np.loadtxt(kwargs["track_subset_file"])
    else: 
        good_frames = None

    def frame_callback(vis, frame_idx, track_class=track_class):
        print("Processing frame %05d" % frame_idx)
        # this is is what should be called detections
        IS_MASK = True
        if IS_MASK:
            detections = create_mask_detections(
                seq_info["detections"], frame_idx, min_detection_height, track_class)
        else:
            detections = create_detections(
                seq_info["detections"], frame_idx, min_detection_height)
        
        # this is the high confidence detections
        high_confidence_detections = [d for d in detections if d.confidence >= min_confidence]
        # These are the low confidences ones and we don't need to run NMS on them because the goal is to retain as much information as possible
        # these should be cmpletely disjoint
        low_confidence_detections = [d for d in detections if d.confidence < min_confidence]

        #HACK temporarily set the value of detections to None to make sure it's not used
        detections = None

        # Run non-maxima suppression.
        # these should only be from high conf detections
        # hc = high_conf
        hc_boxes = np.array([d.tlwh for d in high_confidence_detections])
        hc_scores = np.array([d.confidence for d in high_confidence_detections])
        
        #TODO try to do mask NMS
        indices = preprocessing.non_max_suppression(
            hc_boxes, nms_max_overlap, hc_scores)

        hc_nms_positive_detections = [high_confidence_detections[i] for i in indices] # I think you can just do this by indexing
        # this should negate the value from the line above
        # there might be a cleaner way to do this with sets
        hc_nms_negative_detections = [high_confidence_detections[i] for i in range(len(high_confidence_detections)) if i not in indices]
        assert len(hc_nms_positive_detections) + len(hc_nms_negative_detections) == len(high_confidence_detections), "This operation should just be partitioning detections into two subsets"

        # Update tracker.
        tracker.predict()
        # read the next image because we will actually be using it now
        image = cv2.imread(
            seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
        assert image is not None, "the image now needs to be properly read"
        if stock:
            tracker.update(hc_nms_positive_detections)
        else:
            tracker.update(hc_nms_positive_detections, bad_detections=hc_nms_negative_detections+low_confidence_detections, image=image, use_unmatched=kwargs["use_unmatched"], frame_idx=frame_idx)

        # Update visualization.
        if display:
            vis.set_image(image.copy())
            if seq_info['groundtruth'] is not None:
                vis.draw_groundtruth(*create_groundtruth(seq_info['groundtruth'], frame_idx))
            #HACK for showing detections
            #vis.draw_detections(hc_nms_positive_detections)
            vis.draw_trackers(tracker.tracks, create_groundtruth(seq_info['groundtruth'], frame_idx)) # clean up the double use of create_groundtruht

        # Store results.
        for track in tracker.tracks:
            #MOD write all the tracks, even if not detected recently
            if not track.is_confirmed():# or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

    # Run tracker.
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=5, video_output_file=kwargs["video_output_file"], vis_method=kwargs["vis_method"])
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback, good_frames)

    # Store results.
    f = open(output_file, 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]),file=f)

    # score results
    if seq_info['groundtruth'] is not None:
        scorer = Scorer()
        # I think the groundtruths might be off by a factor of 2
        scores = scorer.score_lists(results, seq_info['groundtruth'], good_frames)
        assert type(scores) ==  pd.DataFrame, "The results are supposed to be a dataframe"
        last_slash_idx = output_file.rfind("/")
        scores_output_file = "{}scores{}".format(output_file[:last_slash_idx + 1], output_file[last_slash_idx + 1:])
        argv_output_file = "{}argv{}".format(output_file[:last_slash_idx + 1], output_file[last_slash_idx + 1:])
        # write the scores to a (very short) file
        current_name = scores.index[0]
        argv = re.sub("[',]", "", str(kwargs["argv"]))
        scores = scores.rename({current_name : argv})
        scores.to_csv(scores_output_file)
        with open(argv_output_file, 'w') as outfile:
            outfile.write(str(kwargs["argv"]))
    else:
        print("There are no groundtruths so no scoring can be done")
        last_slash_idx = output_file.rfind("/")
        argv_output_file = "{}argv{}".format(output_file[:last_slash_idx + 1], output_file[last_slash_idx + 1:])
        with open(argv_output_file, 'w') as outfile:
            outfile.write(str(kwargs["argv"]))



def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default=None, required=True)
    parser.add_argument(
        "--detection_file", help="Path to custom detections.", default=None,
        required=True)
    parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="/tmp/hypotheses.txt")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.8, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None)
    parser.add_argument('--display', default=True, action='store_true', help='Show intermediate tracking results')
    #parser.add_argument(
    #    "--display", help="Show intermediate tracking results",
    #    default=False, action='store_true', type=bool)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args.display)
    run(
        args.sequence_dir, args.detection_file, args.output_file,
        args.min_confidence, args.nms_max_overlap, args.min_detection_height,
        args.max_cosine_distance, args.nn_budget, args.display)
