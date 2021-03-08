# Example Data Preprocessing for Waymo SOT

This folder contains all the related code for preprocessing the Waymo tfrecords for our SOT tracker.

Suppose we want to store the data in `ROOT_DIR`, we have to run the following commands. In the commands, we may use `--process` to split the dataset and accelerate the data preparation.

```bash
python time_stamp.py --data_folder tfrecords_directory --out_folder ROOT_DIR                  # the time stamp of each frame
python ego_info.py --data_folder tfrecords_directory --out_folder ROOT_DIR --process proc_num # the ego information (a 4-by-4 matrix) of each frame
python raw_pc.py --data_folder tfrecords_directory --out_folder ROOT_DIR --process proc_num   # the point cloud on each frame
python gt_info.py --data_folder tfrecords_directory --out_folder ROOT_DIR --process proc_num  # extract the gt bboxes information
python ground_removal.py --data_folder ROOT_DIR --process proc_num                            # remove the ground
python terrain_map.py --data_folder ROOT_DIR                                                  # overlay the point cloud into a terrain map according to ego information
```

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

The only requirements of the codes include
```
waymo_open_dataset: To decode the tfrecords
numpy             : To make the computations
numba             : Accelerate some numpy operations
```
For detailed description and arguments of each file, please look at the annotations on each file.