#!/home/drussel1/anaconda3/bin/zsh
conf=.0
stock='' #this should be --stock is stock and '' if my modifications
nms_thresh='0.7'
name='testing_strong' 
log_file='strong_test_log_desc.txt'
#. activate handtracking >> ${log_file} 2>&1
date# >> ${log_file}
echo name: ${name} conf:${conf} stock: ${stock} nms_thresh: ${nms_thresh} #>> ${log_file}
python evaluate_motchallenge.py --mot_dir MOT16/ADL --detection_dir resources/detections/ADL --output_dir outputs/ADL/tracks/tracks-${name}-${stock}-conf${conf}-nms${nms_thresh} --min_confidence ${conf} --nms_max_overlap ${nms_thresh} ${stock} #>> ${log_file} 2>&1

python my_visualize.py --tracks_dir outputs/ADL/tracks/tracks-${name}-${stock}-conf${conf}-nms${nms_thresh} --image_dir MOT16/ADL --output_dir outputs/ADL/visualizations --MOT_style_images #>> ${log_file} 2>&1
