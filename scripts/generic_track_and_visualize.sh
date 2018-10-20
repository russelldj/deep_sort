#!/home/drussel1/anaconda3/bin/zsh
#CONFIGS##########################################
conf=.01
stock='' #this should be --stock is stock and '' if my modifications
NMS_THRESH='--nms_max_overlap 0.5'
name='4000-6000-ADL' 
track_class='--track_class -1'
MOT_DIR="/home/drussel1/dev/deep_sort/MOT16/ADL0102"
DETECTION_DIR="/home/drussel1/dev/deep_sort/resources/descriptor_detections/ADL/4000-6000"
DATASET="ADL"
VISUALIZE_FRAMES_FILE=''#'--visualize_frames_file /home/drussel1/dev/deep_sort/resources/visualization_frame_flags/EPIC5.txt'
MAX_COSINE_DISTANCE='--max_cosine_distance .4'
MIN_IOU_OVERLAP='--min_iou_overlap .1'
PDB='-m pdb'
log_file=">> ${name}.txt 2>&1"
log_file=''
tracks_dir="outputs/${DATASET}/tracks/tracks-${name}-${stock}-conf${conf}-nms${nms_thresh}"
####################################################

source activate handtracking

date ${log_file}
echo name: ${name} conf:${conf} stock: ${stock} nms_thresh: ${nms_thresh} ${log_file}
eval_line="python ${PDB} evaluate_motchallenge.py --mot_dir ${MOT_DIR} --detection_dir ${DETECTION_DIR} --output_dir ${tracks_dir} --min_confidence ${conf} ${MAX_COSINE_DISTANCE} ${NMS_THRESH} ${MIN_IOU_OVERLAP} ${stock} ${track_class} ${log_file}"
echo ${eval_line}
#eval ${eval_line}

eval "python my_visualize.py --tracks_dir ${tracks_dir} --image_dir ${MOT_DIR} --output_dir outputs/${DATASET}/visualizations --MOT_style_images --conf_thresh 0.3  ${VISUALIZE_FRAMES_FILE} ${log_file}"
