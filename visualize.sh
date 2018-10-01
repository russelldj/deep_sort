conf=.5
source activate handtracking
python --version
python my_visualize.py --output_dir /home/drussel1/dev/deep_sort/custom/visualized_outputs/conf_${conf}  --tracks_dir ./custom/custom_results/conf_${conf} --image_dir ./MOT16/custom_80 
