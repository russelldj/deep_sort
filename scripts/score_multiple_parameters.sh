#!/home/drussel1/anaconda3/bin/zsh
#CONFIGS##########################################
declare -a CONFS=(".1" ".3" ".6")
declare -a MIN_IOU_OVERLAPS=("0.05" "0.15" "0.4")
#log_file=''
####################################################

source activate handtracking

for conf in ${CONFS};
    do 
    for min_iou_overlap in ${MIN_IOU_OVERLAPS}

        do echo $conf ${min_iou_overlaps}
        stock='' #this should be --stock is stock and '' if my modifications
        NMS_THRESH='0.5'
        NAME="all-classes-ADL1620"
        #TRACK_CLASS='--track_class 15'
        MOT_DIR="/home/drussel1/dev/deep_sort/MOT16/ADL18"
        DETECTION_DIR="/home/drussel1/dev/deep_sort/resources/descriptor_detections/ADL_18"
        DATASET="ADL"
        VISUALIZE_FRAMES_FILE='--visualize_frames_file /home/drussel1/dev/deep_sort/resources/visualization_frame_flags/ADL/ADL18.txt'
        MAX_COSINE_DISTANCE='--max_cosine_distance .4'
        PDB="" #'-m pdb'

        tracks_dir="outputs/${DATASET}/tracks/tracks-${NAME}-${stock}-conf${conf}-nms${NMS_THRESH}-iou${min_iou_overlap}"
        log_file="" # ">> score_stock_ADL.txt 2>&1" #">> ${NAME}.txt 2>&1"
        #date ${log_file}
        echo name: ${NAME} conf:${conf} stock: ${stock} NMS_THRESH: ${nms_thresh} ${log_file}
        track_line="python ${PDB} evaluate_motchallenge.py --mot_dir ${MOT_DIR} --detection_dir ${DETECTION_DIR} --output_dir ${tracks_dir} --min_confidence ${conf} ${MAX_COSINE_DISTANCE} --nms_max_overlap ${NMS_THRESH} --min_iou_overlap ${min_iou_overlap} ${stock} ${TRACK_CLASS} ${log_file}"

        #echo ${track_line}
        #eval ${track_line}
        
        vis_line="python my_visualize.py --tracks_dir ${tracks_dir} --image_dir ${MOT_DIR} --output_dir outputs/${DATASET}/visualizations --MOT_style_images --conf_thresh 0.3  ${VISUALIZE_FRAMES_FILE} ${log_file}"
        #echo ${vis_line}
        #eval ${vis_line}
        
        score_line="python /home/drussel1/dev/handtracking/py-motmetrics/motmetrics/apps/eval_motchallenge.py /home/drussel1/data/readonly/MOT_style_gt/ADL ${tracks_dir} ${log_file}"
        echo ${score_line}
        eval ${score_line}

    done
done
