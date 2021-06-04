""" The evaluation code for the output from main.py
    Key functions are:
    tracklet_acc: compute the accuracy of a tracklet
    tracklet_rob: compute the robustness of a tracklet
    dataset_summarize: merge the results of tracklets in the dataset
"""
import os, argparse, json, numpy as np, matplotlib.pyplot as plt, multiprocessing
from copy import deepcopy
import sot_3d
from sot_3d.data_protos import BBox


parser = argparse.ArgumentParser()
parser.add_argument('--name', default='debug', type=str, help='name of the experiment')
parser.add_argument('--result_folder', type=str, default='../../TrackingResults/',
    help='path to store the tracking results')
parser.add_argument('--data_folder', type=str, default='../../datasets/waymo/sot/',
    help='store the data')
parser.add_argument('--bench_list_folder', type=str, default='../benchmark/vehicle', 
    help='the path of benchmark object list')
parser.add_argument('--iou', action='store_true', default=False, help='compute iou yet?')
parser.add_argument('--merge', action='store_true', default=False, help='all the instances at once')
parser.add_argument('--process', type=int, default=1)
args = parser.parse_args()


ROB_THRESHOLDS = np.arange(0, 1.01, 0.05)


def str_list_to_int(lst):
    result = []
    for t in lst:
        try:
            t = int(t)
            result.append(t)
        except:
            continue
    return result


def find_gt_bboxes(id, data_folder, segment_name, start_frame, end_frame):
    """ This function is for finding the gt bboxes
    Args:
        id (str): id of the tracklet
        gt_folder (str): root for data storage
        segment_name (str): the segment to look at
        start_frame (int): which frame to search
    Return
        list of bboxes
    """
    gt_info = np.load(os.path.join(data_folder, 'gt_info', '{:}.npz'.format(segment_name)), 
        allow_pickle=True)
    ego_info = np.load(os.path.join(data_folder, 'ego_info', '{:}.npz'.format(segment_name)), 
        allow_pickle=True)
    bboxes, ids = gt_info['bboxes'][start_frame], gt_info['ids'][start_frame]
    index = ids.index(id)

    gts = list()
    for i in range(start_frame, end_frame + 1):
        # first frame needs no evaluation
        # so begin from start_frame + 1
        frame_bboxes = gt_info['bboxes'][i]
        frame_ids = gt_info['ids'][i]
        index = frame_ids.index(id)
        bbox = frame_bboxes[index]
        bbox = BBox.array2bbox(bbox)

        ego_matrix = ego_info[str(i)]
        bbox = BBox.bbox2world(ego_matrix, bbox)
        gts.append(bbox)

    return gts


def track_len_ratio(ious, threshold):
    """ the ratio of successful tracking, for computing robustness
    """
    track_len = -1
    for i, iou in enumerate(ious):
        if iou < threshold:
            track_len = i + 1
            break
    if track_len == -1:
        track_len = len(ious)
    return track_len / len(ious)


def tracklet_acc(ious):
    """ the accuracy for a tracklet
    """
    return np.average(np.asarray(ious))


def tracklet_rob(ious, thresholds):
    """ compute the robustness of a tracklet
    """
    def compute_area(values):
        """ compute the approximate integral
        """
        area = sum(values[1:-1])
        area = area + (values[0] + values[-1]) / 2
        area *= 0.05
        return area
    robustness = np.zeros(len(thresholds))
    for i, threshold in enumerate(thresholds):
        robustness[i] = track_len_ratio(ious, threshold)
    rob = compute_area(robustness)
    return rob


def metrics_from_bboxes(pred_bboxes, gts):
    """ Compute the accuracy and robustness of a tracklet
    Args:
        pred_bboxes (list of BBox)
        gts (list of BBox)
    Return:
        accuracy, robustness, length of tracklet
    """
    ious = list()
    for pred, gt in zip(pred_bboxes, gts):
        iou3d = sot_3d.utils.iou3d(pred, gt)[1]
        ious.append(iou3d)
    return metrics_from_ious(ious)


def metrics_from_ious(ious):
    """ When we already compute the iou, compute the metrics
    Args:
        ious (list of float)
    Return:
        accuracy, robustness, length of tracklet
    """
    acc = tracklet_acc(ious)
    rob = tracklet_rob(ious, ROB_THRESHOLDS)
    length = len(ious)
    return acc, rob, length


def dataset_summarize(metric_list):
    """ merge the results of multiple tracklets
    Args:
        metric_list: [
            [acc, rob, length], # tracklet 1
            [acc, rob, length], # tracklet 2
            ...
            [acc, rob, length]  # tracklet N
        ]
    Returns:
        accuracy, robustness
    """
    accs, robs, lengths = 0, 0, 0
    for metric in metric_list:
        accs += metric[2] * metric[0]
        robs += metric[2] * metric[1]
        lengths += metric[2]
    return accs / lengths, robs / lengths


def sequence_eval(obj_list, result_folder, name, data_folder, iou, token=0, process=1):
    """ evaluate and return the tracklet-level information in tracklets
    """
    results = list()
    for index, tracklet_info in enumerate(obj_list):
        if index % process != token:
            continue
        tracklet_results = json.load(open(
            os.path.join(result_folder, name, 'summary', '{:}.json'.format(tracklet_info['id']))))
        
        frame_keys = list(tracklet_results.keys())
        frame_keys = sorted(str_list_to_int(frame_keys))
        
        if iou:
            # iou has been computed
            ious = list()
            for key in frame_keys:
                ious.append(tracklet_results[str(key)]['iou3d'])
            tracklet_metrics = metrics_from_ious(ious)
        else:
            gts = find_gt_bboxes(id=tracklet_info['id'],
                                data_folder=data_folder,
                                segment_name=tracklet_info['segment_name'],
                                start_frame=tracklet_info['frame_range'][0] + 1,
                                end_frame=tracklet_info['frame_range'][1])
            preds = list()
            for key in frame_keys:
                bbox = tracklet_results[str(key)]['bbox1']
                bbox = BBox.array2bbox(bbox)
                preds.append(bbox)
            tracklet_metrics = metrics_from_bboxes(preds, gts)
        results.append(tracklet_metrics)
    return results


def main(bench_list_folder, result_folder, name, data_folder, iou):
    if args.merge:
        obj_list = json.load(open(os.path.join(bench_list_folder, 'bench_list.json'), 'r'))
        pool = multiprocessing.Pool(args.process)
        results = pool.starmap(sequence_eval, [
            (obj_list, result_folder, name, data_folder, iou, token, args.process)
            for token in range(args.process)  
        ])
        pool.close()
        pool.join()
        final_results = list()
        for i in range(len(results)):
            for j in range(len(results[i])):
                final_results.append(results[i][j])
        acc, rob = dataset_summarize(final_results)
        print('All set\t -- Acc Rob: {:} {:}'.format(acc, rob))
        return

    hardness_levels = ['easy', 'medium', 'hard']
    for hardness_level in hardness_levels:
        obj_list = json.load(open(os.path.join(bench_list_folder, '{:}.json'.format(hardness_level)), 'r'))
        pool = multiprocessing.Pool(args.process)
        results = pool.starmap(sequence_eval, [
            (obj_list, result_folder, name, data_folder, iou, token, args.process)
            for token in range(args.process)  
        ])
        pool.close()
        pool.join()
        final_results = list()
        for i in range(len(results)):
            for j in range(len(results[i])):
                final_results.append(results[i][j])
        acc, rob = dataset_summarize(final_results)
        print('{:} set\t -- Acc Rob: {:} {:}'.format(hardness_level, acc, rob))
    return


if __name__ == '__main__':
    main(args.bench_list_folder, args.result_folder, args.name, args.data_folder, args.iou)