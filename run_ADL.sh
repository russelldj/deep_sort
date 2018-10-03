python deep_sort_app.py \
        --sequence_dir=./MOT16/custom_80/0000 --detection_file=./resources/detections/custom_80/0000.mp4.npy \
        --min_confidence=0.3 \
        --nn_budget=100 \
        --output_file=results.txt

#scoring I think
#python py-motmetrics/motmetrics/apps/eval_motchallenge.py /home/drussel1/dev/handtracking/deep_sort/MOT16/custom_80 /home/drussel1/dev/handtracking/deep_sort/custom_results
