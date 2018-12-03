# vim: expandtab:ts=4:sw=4
import argparse
import os
import glob
import sys
import deep_sort_app
from deep_sort import tools

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="MOTChallenge evaluation")
    parser.add_argument(
        "--mot-dir", help="Path to MOTChallenge directory (train or test)",
        required=True)
    parser.add_argument(
        "--detection-dir", help="Path to detections.", default="detections",
        required=True)
    parser.add_argument(
        "--output-dir", help="Folder in which the results will be stored. Will "
        "be created if it does not exist.", default="results")
    parser.add_argument(
        "--min-confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.0, type=float)
    parser.add_argument(
        "--min-detection-height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms-max-overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max-cosine-distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn-budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None)
    parser.add_argument(
        "--stock", help="Remove all of the changes I made by using the stock tracker "
        , default=False, action='store_true')
    parser.add_argument(
        "--track-class", help="Only produce tracks for this one class. The expected format is the integer clss label"
        , default=None, type=int)
    parser.add_argument(
        "--min-iou-overlap", help="The minimum IOU required to have a feasible assignment"
        , default=.3, type=float)
    parser.add_argument(
        "--max-age", help="Maximum number of frames a track will be propogated without a detection ", type=int, default=30)
    parser.add_argument("--track-subset-file", help="much the same as the visualization script, this file should be the frame IDs you want tracked, one per line of a file", default=None)
    parser.add_argument("--dont-display", help="use the visualization", action="store_true", default=False)
    parser.add_argument("--dont-use-unmatched", help="don't use unmatched, low-confidence, and NMS-suppressed detections", action="store_false", default=True) # This is really use-unmatched
    parser.add_argument("--tracker-type", help="which tracker to use, currently 'deep-sort', 'flow-matcher', 'flow-tracker'", type=str, default="deep-sort")
    parser.add_argument("--vis-method", help="pick how to visualize the resultant tracks: `show-all`, `one-track`, `one-gt'", type=str, default="one-track")
    parser.add_argument("--flow-dir", help="Where the flows are located", type=str, default="--flow dir should have been set")
    parser.add_argument("--update-kf", help="update the kalman filter to keep the probablilty mass small", action="store_true", default=False)
    parser.add_argument("--update-hit", help="update hit so tracks don't die until they leave the scence", action="store_true", default=False)
    return parser.parse_args()

if __name__ == "__main__":

    #tools.pdb_on_ctrl_c()
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sequences = os.listdir(args.mot_dir)
    for sequence in sequences:
        sequence_dir = os.path.join(args.mot_dir, sequence)
        
        # this is a bit janky but it should deal with the different file extensions
        detection_file = os.path.join(args.detection_dir, sequence)
        detection_file = glob.glob("{}*".format(detection_file)) 
        assert len(detection_file) == 1
        detection_file = detection_file[0]
        print("Running sequence {} with detection file {}".format(sequence, detection_file))

        output_file = os.path.join(args.output_dir, "%s.txt" % sequence)
        video_output_file = os.path.join(args.output_dir, "{}_video.avi".format(sequence))
        deep_sort_app.run(
            sequence_dir, detection_file, output_file, args.min_confidence,
            args.nms_max_overlap, args.min_detection_height,
            args.max_cosine_distance, args.nn_budget, display=(not args.dont_display), stock=args.stock, track_class=args.track_class, max_age=args.max_age, min_iou_overlap=args.min_iou_overlap, track_subset_file=args.track_subset_file, use_unmatched=args.dont_use_unmatched, video_output_file=video_output_file, tracker_type=args.tracker_type, vis_method=args.vis_method, argv=sys.argv, flow_dir=args.flow_dir, update_hit=args.update_hit, update_kf=args.update_kf)
