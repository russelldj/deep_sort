import motmetrics as mm
from collections import defaultdict, namedtuple
import numpy as np
import time

class Scorer(object):
    def __init__(self):
        self.acc = mm.MOTAccumulator(auto_id=True)
        pass

    def score_lists(self, tracks, groundtruths, frame_subset=None, name="Some sequence"):
        """ Take natively-formated tracks and groundtruths and return the string representation of the MOT metric results

        Parameters
        ---------- 
        tracks : List[ List[ float ] ]
            The list of tracks in the format [frame_id, track_id, top, left, width, height]
        groundtruths : numpy.ndarray
            shape should be (n, 10) where each line is in the MOT format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

            >>> scorer = Scorer()
            ... import numpy as np
            ... tracks = [[15, 1, 0, 0, 100, 100], [30, 1, 100, 100, 100, 100]]
            ... gts = np.array([[0, 2, 0, 0, 100, 100, 1, -1, -1, -1], [30, 2, 100, 100, 50, 100, 1, -1, -1, -1]])
            ... summary = scorer.score_lists(tracks, gts, frame_subset=[0, 30])
            ... assert summary["num_frames"][0] == 2
            ... assert summary["mota"][0] == 0.5
            ... assert summary["motp"][0] == 0.5

        """
        
        # need to get all of the frames in the groundtruth, and only evaluate those
        # For now, I think we can ignore the cases where there a frames which should be groundtruths but doesn't have annotations, though if they appear to all be on multiples of 30, then I'll reconsider
        print("Began computing scores")
        gt_frames = groundtruths[:, 0]

        if frame_subset is not None:
            gt_frames = [gtf for gtf in gt_frames if gtf in frame_subset]

        tracks = [track for track in tracks if track[0] in gt_frames]
      
        # perhaps poorly-named but I couldn't think of anything better
        # frame is somewhat reduntant here
        TrackFrame = namedtuple("TrackFrame", ["frame", "ID", "bbox"])

        tracks_dict = defaultdict(list)
        gts_dict     = defaultdict(list)

        #perhaps the conversion should be performed here
        # they provide a module to do IOU distancing, so it should probably put in that format
        for track in tracks:
            track_frame = TrackFrame(frame=track[0], ID=track[1], bbox=np.array(track[2:6]))
            tracks_dict[track_frame.frame].append(track_frame)

        for gt in groundtruths:
            gt_frame = TrackFrame(frame=gt[0], ID=gt[1], bbox=gt[2:6])
            gts_dict[gt_frame.frame].append(gt_frame)
        
        print("Cleaned the data")

        for frame in gt_frames:
            tracks = np.array([t.bbox for t in tracks_dict[frame]])
            gts    = np.array([g.bbox for g in gts_dict[frame]])
            
            dists = mm.distances.iou_matrix(gts, tracks) 
            track_ids = [t.ID for t in tracks_dict[frame]]
            gt_ids    = [str(g.ID) for g in gts_dict[frame]]
            frameid = self.acc.update(gt_ids, track_ids, dists)

        print("Finished accumulating the data")

        mh = mm.metrics.create()
        start = time.time()
        summary = mh.compute(self.acc, metrics=mm.metrics.motchallenge_metrics, name=name)
        end = time.time()
        print("Computed the summary statistics in {} seconds".format(end - start))
        print(summary)
        return summary


