# Example Data Preprocessing for Waymo SOT

This folder contains all the related code for preprocessing the Waymo tfrecords for our SOT tracker.

## 1. Commands

For a quick data preprocessing, run the following command.

```bash
cd waymo_data/data_preprocessing
bash preprocessing_pipeline.sh tfrecords_directory root_dir process_number
# tfrecords_directory: the directory to store the tfrecords of the validation sets of WOD
# root_dir:            the directory to store your preprocessed files
# process_number:      the process number to use, accelerating the preprocessing
```

A split of the commands are:

```bash
python time_stamp.py --data_folder tfrecords_directory --output_folder ROOT_DIR                  # the time stamp of each frame
python ego_info.py --data_folder tfrecords_directory --output_folder ROOT_DIR --process proc_num # the ego information (a 4-by-4 matrix) of each frame
python raw_pc.py --data_folder tfrecords_directory --output_folder ROOT_DIR --process proc_num      # the point cloud on each frame
python ground_removal.py --data_folder ROOT_DIR --process proc_num                               # remove the ground
python gt_info.py --data_folder tfrecords_directory --output_folder ROOT_DIR --process proc_num     # extract the gt bboxes information
python terrain_map.py --data_folder ROOT_DIR --process proc_num                                  # overlay the point cloud into a terrain map according to ego information
```

## 2. Output Files' Structure
```
--- ROOT_DIRECTORY
    --- ts_info:
        time stampe information for each segment
    --- pc:
        --- raw_pc:
            raw point cloud data. details in raw_pc.py
        --- clean_pc:
            point cloud after ground removal. details in ground_removal.py
        --- ground_pc:
            ground points. details in ground_removal.py
        --- terrain:
            a pseudo terrain map. details in terrain_map.py
    --- ego_info:
        the pose matrix. in ego_info.py
    --- gt_info:
        the gt bboxes, their types, ids and point numbers. in gt_info.py
```

## 3. Requirements

The only requirements of the codes include

```
waymo_open_dataset: To decode the tfrecords
numpy             : To make the computations
numba             : Accelerate some numpy operations
```
For detailed description and arguments of each file, please look at the annotations on each file.
