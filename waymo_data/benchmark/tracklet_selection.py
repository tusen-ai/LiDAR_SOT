""" Selecting the proper tracklets in the following steps:
    1. Reformat each gt sequence into individual tracklets, only keep the vehicle ones
    2. Move the start of each tracklet, so that their beginning point cloud number is proper
    3. Filter the short sequences
    4. Select the mobile objects by a distance threshold
    5. Compute the starting LiDAR point number, removing the improper ones and divide into groups
"""
import numpy as np, os, argparse, json
from seq2tracklet import seq2tracklets
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--gt_info_folder', type=str, default='/mnt/truenas/scratch/ziqi.pang/datasets/waymo/sot/gt_info/')
parser.add_argument('--ego_info_folder', type=str, default='/mnt/truenas/scratch/ziqi.pang/datasets/waymo/sot/ego_info/')
parser.add_argument('--bench_info_folder', type=str, default='../../benchmark/')
parser.add_argument('--obj_type', type=str, default='vehicle')
args = parser.parse_args()


type_dict = {
    'vehicle': 1,
    'pedestrian': 2,
    'unknown': 0,
    'cyclist': 4
}


def move_start_of_tracklet(tracklets, prep_len=10, pc_num_thres=20):
    """ Move the start frame of tracklet.
        When the beginning frame do not have enough LiDAR points, move it back
    Args:
        tracklets ([Tracklet]): input tracklets 
        pc_num_thres: least number of LiDAR points
    Return:
        list of tracklets after moving the start
    """
    result = list()
    for tracklet_index, tracklet in enumerate(tracklets):
        pc_nums = tracklet.pc_nums
        # find the valid start
        valid_start = -1
        for i in range(len(pc_nums)-prep_len):
            if min(pc_nums[i:i+prep_len]) >= pc_num_thres:
                valid_start = i
                break
        
        # move the tracklet
        if valid_start == -1:
            continue
        tracklet.start_frame += valid_start
        tracklet.pc_nums = tracklet.pc_nums[valid_start:]
        tracklet.bboxes = tracklet.bboxes[valid_start:]
        result.append(tracklet)
    return result       


def filter_by_frame_range(tracklets, min_len=100):
    """ Filter the tracklets not long enough
    Args:
        tracklets ([Tracklet]): list of tracklets
    Return:
        list of tracklets satisfying the length
    """
    result = list()
    for tracklet_index, tracklet in enumerate(tracklets):
        tracklet_len = tracklet.end_frame - tracklet.start_frame + 1
        if tracklet_len >= min_len:
            result.append(tracklet)
    return result


def filter_by_moving(tracklets, ego_info, distance_threshold=1.0):
    """ Compute if the starting and ending location of the object is larger than the distance threshold.

    Args:
        tracklets ([Tracklets]): list of tracklets
        ego_info (dict of ego_matrices)
        distance_threshold (float): distance threshold
    Return:
        Proper tracklets
    """
    result = list()
    for tracklet in tracklets:
        start_bbox, end_bbox = tracklet.bboxes[0], tracklet.bboxes[-1]
        start_ego, end_ego = ego_info[str(tracklet.start_frame)], ego_info[str(tracklet.end_frame)]
        start_bbox = bbox2world(start_ego, start_bbox)
        end_bbox = bbox2world(end_ego, end_bbox)
        dist = np.linalg.norm((end_bbox - start_bbox)[:3])

        if dist > distance_threshold:
            result.append(tracklet)
    return result


def divide_into_groups(tracklets, prep_len=10):
    """ Divide the tracklets into different groups according to their LiDAR point number in the first frames
    Args:
        tracklets (list of tracklets): input tracklet information
        prep_len (int, optional): how many frames to concern in the beginning of tracklets
    Return:
        [[easy tracklets], [medium tracklets], [hard lists]]
    """
    avg_pc_nums = list()
    for tracklet in tracklets:
        val = np.average(np.asarray(tracklet.pc_nums[:prep_len]))
        avg_pc_nums.append(val)
    
    indices = np.argsort(avg_pc_nums)
    tracklet_num = len(indices)

    # indices and tracklets of easy, medium, hard
    group_indices = [indices[:tracklet_num // 3], indices[tracklet_num // 3: tracklet_num * 2 // 3], indices[tracklet_num * 2 // 3 :]]
    group_lists = [[] for i in range(3)]
    for i, indices in enumerate(group_indices):
        for j in indices:
            group_lists[i].append(tracklets[j])
    return group_lists


def main(obj_type, gt_info_folder, ego_info_folder, bench_info_folder):
    num_obj_type = type_dict[obj_type]
    file_names = sorted(os.listdir(gt_info_folder))
    result_tracklets = list()
    for file_index, file_name in enumerate(file_names):
        print('START {:} / {:}'.format(file_index + 1, len(file_names)))
        segment_name = file_name.split('.')[0]
        gt_data = np.load(os.path.join(gt_info_folder, file_name), allow_pickle=True)
        ego_data = np.load(os.path.join(ego_info_folder, file_name), allow_pickle=True)
        bboxes, ids, obj_types, pc_nums = gt_data['bboxes'], gt_data['ids'], gt_data['types'], gt_data['pc_nums']
        
        _, tracklets = seq2tracklets(bboxes, ids, obj_types, pc_nums, [num_obj_type])
        tracklets = move_start_of_tracklet(tracklets)
        tracklets = filter_by_frame_range(tracklets)
        tracklets = filter_by_moving(tracklets, ego_data)

        for tracklet in tracklets:
            tracklet.segment_name = segment_name
            result_tracklets.append(tracklet)
    print('{:} tracklets selected.'.format(len(result_tracklets)))

    group_tracklets = divide_into_groups(result_tracklets)
    difficulty_level = ['hard', 'medium', 'easy']
    group_result = [[] for i in range(3)]
    for i, (level_name, level_tracklets) in enumerate(zip(difficulty_level, group_tracklets)):
        for tracklet in level_tracklets:
            tracklet_info = tracklet.tracklet2dict()
            group_result[i].append(tracklet_info)
        f = open(os.path.join(bench_info_folder, obj_type, '{:}.json'.format(level_name)), 'w')
        json.dump(group_result[i], f)
        f.close()
    
    all_list = group_result[2] + group_result[1] + group_result[0]
    f = open(os.path.join(bench_info_folder, obj_type, 'bench_list.json'), 'w')
    json.dump(all_list, f)
    f.close()


if __name__ == '__main__':
    main(args.obj_type, args.gt_info_folder, args.ego_info_folder, args.bench_info_folder)
