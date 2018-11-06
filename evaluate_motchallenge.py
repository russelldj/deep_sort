# vim: expandtab:ts=4:sw=4
import argparse
import os
import deep_sort_app
from deep_sort import tools


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="MOTChallenge evaluation")
    parser.add_argument(
        "--mot_dir", help="Path to MOTChallenge directory (train or test)",
        required=True)
    parser.add_argument(
        "--detection_dir", help="Path to detections.", default="detections",
        required=True)
    parser.add_argument(
        "--output_dir", help="Folder in which the results will be stored. Will "
        "be created if it does not exist.", default="results")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.0, type=float)
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
    parser.add_argument(
        "--stock", help="Remove all of the changes I made by using the stock tracker "
        , default=False, action='store_true')
    parser.add_argument(
        "--track_class", help="Only produce tracks for this one class. The expected format is the integer clss label"
        , default=None, type=int)
    parser.add_argument(
        "--min_iou_overlap", help="The minimum IOU required to have a feasible assignment"
        , default=.3, type=float)
    parser.add_argument(
        "--max_age", help="Maximum number of frames a track will be propogated without a detection ", type=int, default=30)
    parser.add_argument("--track_subset_file", help="much the same as the visualization script, this file should be the frame IDs you want tracked, one per line of a file", default=None)
    parser.add_argument("--dont_display", help="use the visualization", action="store_true", default=False)
    parser.add_argument("--use_unmatched", help="use unmatched, low-confidence, and NMS-suppressed detections", action="store_false", default=True)
    parser.add_argument("--video_output_file", help="instead of visualizing the frames write them to this video", type=str, default=None)
    parser.add_argument("--use-flow", help="Use optical flow rather than appearance features", action="store_true", default=False)
    return parser.parse_args()

if __name__ == "__main__":

    tools.pdb_on_ctrl_c()
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sequences = os.listdir(args.mot_dir)
    for sequence in sequences:
        print("Running sequence %s" % sequence)
        sequence_dir = os.path.join(args.mot_dir, sequence)
        detection_file = os.path.join(args.detection_dir, "%s.npy" % sequence)
        output_file = os.path.join(args.output_dir, "%s.txt" % sequence)
        deep_sort_app.run(
            sequence_dir, detection_file, output_file, args.min_confidence,
            args.nms_max_overlap, args.min_detection_height,
            args.max_cosine_distance, args.nn_budget, display=(not args.dont_display), stock=args.stock, track_class=args.track_class, max_age=args.max_age, min_iou_overlap=args.min_iou_overlap, track_subset_file=args.track_subset_file, use_unmatched=args.use_unmatched, video_output_file=args.video_output_file, use_flow=args.use_flow)
