track_folder='stock_conf.5_nms.7'
log_file='stock_scoring_log.txt'
cd /home/drussel1/dev/deep_sort
python ../py-motmetrics/motmetrics/apps/eval_motchallenge.py /home/drussel1/dev/handtracking/deep_sort/MOT16/custom_80 /home/drussel1/dev/handtracking/deep_sort/outputs/custom/tracks/${track_folder} >> ${log_file} 2>&1
