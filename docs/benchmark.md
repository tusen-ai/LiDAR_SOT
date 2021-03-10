# Benchmark Creation

## 1. Benchmark

We have already provided the final result of benchmark in `LiDAR_SOT/benchmark/vehicle/`, where `easy.json`, `medium.json`, `hard.json` and `bench_list.json` are the information about selected tracklets. Inside each json file, we follow the format of:

```
[
    # information about tracklet 0
    {
        'id': id of object 0,
        'type': type of object 0,
        'segment_name': the segment this tracklet is in,
        'frame_range': the number of the first and last frame of this tracklet in the sequence
    },
    # information about tracklet 1
    {
        ...
    },
    ...
    # information about tracklet N
    {
        ...
    }
]
```

## 2. Replicate the Benchmark Creation

`LiDAR_SOT/waymo_data/benchmark/` contains the codes used for selecting the tracklets for our SOT benchmark.  The motivation for providing these code are simply for people to replicate the process of creating our benchmark. The creation of our benchmark has to follow the preparation of waymo validation set data, as described in [Data Processing](./data_preprocessing.md). To get the selected tracklets, run 

```bash
python tracklet_selection.py --data_folder root_dir_in_data_preprocessing
```

After running the code, you could access the tracklets in `LiDAR_SOT/benchmark/`, 



