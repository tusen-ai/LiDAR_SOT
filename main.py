import os, argparse, json, numpy as np, yaml, multiprocessing, shutil
import sot_3d, sot_3d.utils as utils
from sot_3d.data_protos import BBox
from data_loader import ExampleLoader
from copy import deepcopy
from sot_3d.visualization import Visualizer2D


parser = argparse.ArgumentParser()
# paths
parser.add_argument('--bench_list', type=str, default='./benchmark/vehicle/bench_list.json', 
    help='the path of benchmark object list')
parser.add_argument('--data_folder', type=str, default='../datasets/waymo/sot/',
    help='store the data')
parser.add_argument('--result_folder', type=str, default='../TrackingResults/',
    help='path to store the tracking results')
parser.add_argument('--config_path', type=str, default='config.yaml', help='config path')
# running configurations
parser.add_argument('--name', type=str, default='debug', help='name of this experiments')
parser.add_argument('--process', type=int, default=1, help='multiprocessing for acceleration')
parser.add_argument('--skip', action='store_true', default=False, help='skip the tracklets already finish')
parser.add_argument('--visualize', action='store_true', default=False)
# debug mode
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--max_len', type=int, default=200)
# parse arguments
args = parser.parse_args()


def find_bboxes(id, data_folder, segment_name, start_frame, end_frame):
    """ In the SOT, the beginning frame is a GT BBox
        This function is for finding the gt bbox
    Args:
        id (str): id of the tracklet
        data_folder (str): root for data storage
        segment_name (str): the segment to look at
        start_frame (int): which frame to search
    Return
        BBox (numpy array): [x, y, z, h, l, w, h]
    """
    gt_info = np.load(os.path.join(data_folder, 'gt_info', '{:}.npz'.format(segment_name)), 
        allow_pickle=True)
    ego_info = np.load(os.path.join(data_folder, 'ego_info', '{:}.npz'.format(segment_name)), 
        allow_pickle=True)
    bboxes, ids = gt_info['bboxes'][start_frame], gt_info['ids'][start_frame]
    index = ids.index(id)
    start_bbox = bboxes[index]

    gts = list()
    for i in range(start_frame, end_frame + 1):
        frame_bboxes = gt_info['bboxes'][i]
        frame_ids = gt_info['ids'][i]
        index = frame_ids.index(id)
        bbox = frame_bboxes[index]
        bbox = BBox.array2bbox(bbox)

        ego_matrix = ego_info[str(i)]
        bbox = BBox.bbox2world(ego_matrix, bbox)
        gts.append(bbox)

    return start_bbox, gts


def compare_to_gt(cur_frame_idx, frame_result, gts=None):
    """ For the detail of frame_result, refer to the get_frame_result in tracker.py
    """
    result = deepcopy(frame_result)
    max_frame_key = max(list(frame_result.keys()))
    for i in range(max_frame_key + 1):
        frame_idx = cur_frame_idx - (max_frame_key - i)
        bbox0, bbox1 = frame_result[i]['bbox0'], frame_result[i]['bbox1']
        result[i]['bbox0'] = BBox.bbox2array(bbox0).tolist()
        result[i]['bbox1'] = BBox.bbox2array(bbox1).tolist()

        if gts:
            gt_bbox0, gt_bbox1 = gts[frame_idx - 1], gts[frame_idx]
            iou_2d, iou_3d = sot_3d.utils.iou3d(bbox1, gt_bbox1)
            
            result[i]['gt_bbox0'] = BBox.bbox2array(gt_bbox0).tolist()
            result[i]['gt_bbox1'] = BBox.bbox2array(gt_bbox1).tolist()
            result[i]['gt_motion'] = (BBox.bbox2array(gt_bbox1) - BBox.bbox2array(gt_bbox0))[:4].tolist()
            result[i]['iou2d'] = iou_2d
            result[i]['iou3d'] = iou_3d
    return result


def id_track(configs, id, segment_name, frame_range, data_folder):
    """ ID tracking, prepare the data loader and call the tracker_api
    """
    # initialize the data loader
    data_loader = ExampleLoader(configs=configs, id=id, segment_name=segment_name, 
        data_folder=data_folder, frame_range=frame_range)
    # find the starting bbox
    start_bbox, gts = find_bboxes(id=id, data_folder=data_folder, 
        segment_name=segment_name, start_frame=frame_range[0], end_frame=frame_range[1])
    # run the tracker
    tracking_results = tracker_api(configs=configs, id=id, start_bbox=start_bbox,
        start_frame=frame_range[0], data_loader=data_loader, track_len=frame_range[1]-frame_range[0]+1,
        gts=gts, visualize=args.visualize)
    return tracking_results


def frame_result_visualization(frame_result, pc):
    visualizer = Visualizer2D(figsize=(12, 12))
    bbox0, bbox1 = frame_result['bbox0'], frame_result['bbox1']
    gt_bbox0, gt_bbox1 = frame_result['gt_bbox0'], frame_result['gt_bbox1']
    bbox1, gt_bbox1 = BBox.array2bbox(bbox1), BBox.array2bbox(gt_bbox1)
    visualizer.handler_box(bbox1, color='bbox1')
    visualizer.handler_box(gt_bbox1, color='bbox0')
    vis_pc = utils.pc_in_box_2D(gt_bbox1, pc, 4.0)
    visualizer.handler_pc(vis_pc)
    visualizer.show()
    visualizer.close()


def tracker_api(configs, id, start_bbox, start_frame, data_loader, track_len, gts=None, visualize=False):
    """ api for the tracker
    Args:
        configs: model configuration read from config.yaml
        id (str): each tracklet has an id
        start_bbox ([x, y, z, yaw, l, w, h]): the beginning location of this id
        data_loader (an iterator): iterator returning data of each incoming frame
    Return:
        {
            frame_number0: pred_bbox0,
            frame_number1: pred_bbox1,
            ...
            frame_numberN: pred_bboxN
        }
    """
    tracker = sot_3d.Tracker(id=id, configs=configs, start_bbox=start_bbox, start_frame=start_frame, track_len=track_len)
    tracklet_result = dict()
    for frame_index in range(track_len):
        print('////////////////////////////////////////')
        print('Processing {:} {:} / {:}'.format(id, frame_index + 1, track_len))
        # initialize a tracker
        frame_data = next(data_loader)
        # if the first frame, add the start_bbox
        input_bbox = None
        if frame_index == 0:
            input_bbox = BBox.bbox2world(frame_data['ego'], BBox.array2bbox(start_bbox))
        input_data = sot_3d.FrameData(ego_info=frame_data['ego'], pc=frame_data['pc'], start_bbox=input_bbox,
            terrain=frame_data['terrain'], dets=frame_data['dets'])
        
        # run the frame level tracking
        frame_output = tracker.track(input_data)
        # the frame 0 may produce no output
        if not frame_output:
            continue

        # if gt is not None, we may compare our prediction with gt
        frame_result = compare_to_gt(frame_index, frame_output, gts)
        max_frame_key = max(list(frame_result.keys()))
        for i in range(max_frame_key + 1):
            print('BBox0    : {:}'.format(frame_result[i]['bbox0']))
            print('BBox1    : {:}'.format(frame_result[i]['bbox1']))
            print('Motion   : {:}'.format(frame_result[i]['motion']))
            if gts:
                print('GT BBox0 : {:}'.format(frame_result[i]['gt_bbox0']))
                print('GT BBox1 : {:}'.format(frame_result[i]['gt_bbox1']))
                print('GT Motion: {:}'.format(frame_result[i]['gt_motion']))
                print('IOUS     : {:}  {:}'.format(frame_result[i]['iou2d'], frame_result[i]['iou3d']))
            print('\n')
        tracklet_result[frame_index + start_frame] = frame_result[max_frame_key]

        if visualize:
            frame_result_visualization(frame_result[max_frame_key], tracker.input_data.pc)
    return tracklet_result


def main(name, config_path, bench_list, data_folder, result_folder, token=0, process=1):
    summary_folder = os.path.join(result_folder, 'summary')
    # load configuration file
    f = open(config_path, 'r')
    configs = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    shutil.copy(config_path, os.path.join(result_folder, 'config.yaml'))

    # load bench list
    bench_list = json.load(open(bench_list, 'r'))
    if args.skip:
        final_bench_list = list()
        for tracklet_info in bench_list:
            if not os.path.exists(os.path.join(summary_folder, '{:}.json'.format(tracklet_info['id']))):
                final_bench_list.append(tracklet_info)
    else:
        final_bench_list = bench_list
    
    # iterate over all the objects
    for tracklet_index, tracklet_info in enumerate(final_bench_list):
        if tracklet_index % process != token:
            continue
        print('START ID {:}, {:} / {:}'.format(tracklet_info['id'], tracklet_index + 1, len(final_bench_list)))
        
        frame_range = tracklet_info['frame_range']
        # for the sake of debug
        if args.debug:
            frame_range[1] = min(frame_range[1], frame_range[0] + args.max_len - 1)
        tracking_results = id_track(configs, tracklet_info['id'], tracklet_info['segment_name'], frame_range, data_folder)
        
        tracklet_result_path = os.path.join(summary_folder, '{:}.json'.format(tracklet_info['id']))
        f = open(tracklet_result_path, 'w')
        json.dump(tracking_results, f)
        f.close()


if __name__ == '__main__':
    if not os.path.exists(os.path.join(args.result_folder, args.name)):
        os.makedirs(os.path.join(args.result_folder, args.name))
    result_folder = os.path.join(args.result_folder, args.name)
    summary_folder = os.path.join(result_folder, 'summary')
    if not os.path.exists(os.path.join(summary_folder)):
        os.makedirs(summary_folder)

    if args.process > 1:
        pool = multiprocessing.Pool(args.process)
        for token in range(args.process):
            result = pool.apply_async(main, args=(args.name, args.config_path, args.bench_list, args.data_folder, result_folder, 
                token, args.process))
            # result.get()
        pool.close()
        pool.join()
    else:
        main(args.name, args.config_path, args.bench_list, args.data_folder, result_folder)
