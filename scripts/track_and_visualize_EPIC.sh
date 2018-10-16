#!/home/drussel1/anaconda3/bin/zsh
conf=.0
stock='' #this should be --stock is stock and '' if my modifications
nms_thresh='0.7'
name='5_with_descs' 
log_file='>> 5_with_descs.txt 2>&1'
log_file=''
#. activate handtracking >> ${log_file} 2>&1
date >> ${log_file}
echo name: ${name} conf:${conf} stock: ${stock} nms_thresh: ${nms_thresh} >> log.txt
python evaluate_motchallenge.py --mot_dir resources/tmp/EPIC_5_video/ --detection_dir resources/tmp/EPIC_5_mask_w_descriptors --output_dir outputs/EPIC/tracks/tracks-${name}-${stock}-conf${conf}-nms${nms_thresh} --min_confidence ${conf} --nms_max_overlap ${nms_thresh} ${stock} ${log_file}

python my_visualize.py --tracks_dir outputs/EPIC/tracks/tracks-${name}-${stock}-conf${conf}-nms${nms_thresh} --image_dir resources/tmp/EPIC_5_video/ --output_dir outputs/EPIC/visualizations --MOT_style_images ${log_file}
