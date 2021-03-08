# Benchmark Creation

This folder contains the codes used for selecting the tracklets for our SOT benchmark. People do not need to run the code in the folder, they may directly read the files in `TrackingAndEstimation/benchmark/` for the chosen tracklets. The motivation for providing these code are simply for people to replicate the process of creating our benchmark.

The creation of our benchmark has to follow the preparation of waymo validation set data, as described in [Data Processing](./data_preprocessing.md). To get the selected tracklets, run `python tracklet_selection.py`.

After running the code, you could access the tracklets in `./TrackingAndEstimation/benchmark/`, where `easy.json`, `medium.json`, `hard.json` and `bench_list.json` are the information about selected tracklets. Inside each json file, we follow the format of:
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