conf=.3
stock='' #this should be --stock is stock and '' if my modifications
nms_thresh='0.7'
name='testing_strong' 
log_file='scripts/conf.3_log.txt'
date >> log.txt
echo name: ${name} conf:${conf} stock: ${stock} nms_thresh: ${nms_thresh} >> log.txt
python evaluate_motchallenge.py --mot_dir MOT16/custom_80 --detection_dir resources/detections/custom_80 --output_dir outputs/custom/tracks/tracks-${name}-${stock}-conf${conf}-nms${nms_thresh} --min_confidence ${conf} --nms_max_overlap ${nms_thresh} ${stock} >> ${log_file} 2>&1

python my_visualize.py --tracks_dir outputs/custom/tracks/tracks-${name}-${stock}-conf${conf}-nms${nms_thresh} --image_dir MOT16/custom_80 --output_dir outputs/custom/visualizations --MOT_style_images >> ${log_file} 2>&1
