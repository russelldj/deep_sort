#!/home/drussel1/anaconda3/bin/zsh
#CONFIGS##########################################
declare -a CONFS=(".3")  #(".6" ".3" ".1")
declare -a MIN_IOU_OVERLAPS=("0.05") #("0.05" "0.15" "0.4")
declare -a NMS_THRESHS=("0.05") # ("0.05" "0.15" "0.4")
declare -a MAX_COSINE_DISTANCES=("0.6") #("0.2" "0.4" "0.6" "1.0")
#log_file=''
####################################################

source activate handtracking

for conf in ${CONFS};
    do 
    for NMS_THRESH in ${NMS_THRESHS}
        do
        for min_iou_overlap in ${MIN_IOU_OVERLAPS}
            do
            for max_cosine_distance in ${MAX_COSINE_DISTANCES}
                do echo $conf ${min_iou_overlaps}
                stock='' #this should be --stock is stock and '' if my modifications
                #NMS_THRESH='0.05'
                NAME="ADL_exaustive_search"
                #TRACK_CLASS='--track_class 15'
                MOT_DIR="/home/drussel1/dev/deep_sort/MOT16/ADL18"
                DETECTION_DIR="/home/drussel1/dev/deep_sort/resources/descriptor_detections/ADL_18"
                DATASET="ADL"
                VISUALIZE_FRAMES_FILE='/home/drussel1/dev/deep_sort/resources/visualization_frame_flags/ADL/ADL18.txt'
                MAX_COSINE_DISTANCE="--max_cosine_distance ${max_cosine_distance}"
                PDB='-m pdb'

                tracks_dir="outputs/${DATASET}/tracks/tracks-${NAME}-conf${conf}-nms${NMS_THRESH}-iou${min_iou_overlap}-max_cosine_distance${max_cosine_distance}"
                log_file="" #">> ${NAME}.txt 2>&1"

                date ${log_file}
                #echo name: ${NAME} conf:${conf} stock: ${stock} NMS_THRESH: ${nms_thresh} ${log_file}
                track_line="python ${PDB} evaluate_motchallenge.py --mot_dir ${MOT_DIR} --detection_dir ${DETECTION_DIR} --output_dir ${tracks_dir} --min_confidence ${conf} ${MAX_COSINE_DISTANCE} --nms_max_overlap ${NMS_THRESH} --min_iou_overlap ${min_iou_overlap} ${stock} ${TRACK_CLASS} ${log_file} --track_subset_file ${VISUALIZE_FRAMES_FILE}"
                echo ${track_line}
                #eval ${track_line}
                
                vis_line="python ${PDB} my_visualize.py --tracks_dir ${tracks_dir} --image_dir ${MOT_DIR} --output_dir outputs/${DATASET}/visualizations --MOT_style_images --conf_thresh 0.3 --visualize_frames_file ${VISUALIZE_FRAMES_FILE} ${log_file}"
                echo ${vis_line}
                #eval ${vis_line}
            done
        done
    done
done
