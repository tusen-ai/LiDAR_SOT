tf_dir=$1
root_dir=$2
proc_num=$3
proc_num=$(($proc_num))

python time_stamp.py --data_folder tf_dir --output_folder root_dir 
python ego_info.py --data_folder tf_dir --output_folder root_dir --process proc_num
python raw_pc.py --data_folder tf_dir --output_folder root_dir --process proc_num
python ground_removal.py --data_folder root_dir --process proc_num
python gt_info.py --data_folder tf_dir --output_folder root_dir --process proc_num
python terrain_map.py --data_folder root_dir --process proc_num