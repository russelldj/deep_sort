#!/home/drussel1/anaconda3/bin/zsh
conf=.0
stock='' #this should be --stock is stock and '' if my modifications
nms_thresh='0.7'
name='testing_EPIC5_faster_rcnn_dets_cosine_descs' 
log_file=">> ${name}.txt 2>&1"
log_file=''
MOT_DIR="/home/drussel1/dev/deep_sort/resources/tmp/EPIC_5_video"
DETECTION_DIR="/home/drussel1/dev/deep_sort/resources/descriptor_detections/EPIC_faster_dets_cosine_descs"
#. activate handtracking >> ${log_file} 2>&1
date ${log_file}
echo name: ${name} conf:${conf} stock: ${stock} nms_thresh: ${nms_thresh} ${log_file}
python evaluate_motchallenge.py --mot_dir ${MOT_DIR} --detection_dir ${DETECTION_DIR} --output_dir outputs/EPIC/tracks/tracks-${name}-${stock}-conf${conf}-nms${nms_thresh} --min_confidence ${conf} --nms_max_overlap ${nms_thresh} ${stock} ${log_file}

python my_visualize.py --tracks_dir outputs/EPIC/tracks/tracks-${name}-${stock}-conf${conf}-nms${nms_thresh} --image_dir ${MOT_DIR} --output_dir outputs/EPIC/visualizations --MOT_style_images --conf_thresh 0.3 ${log_file}
