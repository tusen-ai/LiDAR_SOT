""" overlay the gt shapes for shape evaluation
"""
import numpy as np, os, multiprocessing, argparse, json, sot_3d
from sot_3d.data_protos import BBox


parser = argparse.ArgumentParser()
parser.add_argument('--bench_list', type=str, default='../../benchmark/vehicle/bench_list.json', 
    help='the path of benchmark object list')
parser.add_argument('--data_folder', type=str, default='../../../datasets/waymo/sot/')
parser.add_argument('--process', type=int, default=1)
args = parser.parse_args()


def find_bboxes(id, data_folder, segment_name, start_frame, end_frame):
    """ This function is for finding the gt bboxes
    """
    gt_info = np.load(os.path.join(data_folder, 'gt_info', '{:}.npz'.format(segment_name)), 
        allow_pickle=True)
    bboxes, ids = gt_info['bboxes'][start_frame], gt_info['ids'][start_frame]
    index = ids.index(id)

    gts = list()
    for i in range(start_frame, end_frame + 1):
        frame_bboxes = gt_info['bboxes'][i]
        frame_ids = gt_info['ids'][i]
        index = frame_ids.index(id)
        bbox = frame_bboxes[index]
        bbox = BBox.array2bbox(bbox)
        gts.append(bbox)
    return gts


def create_gt_shape(pcs, bboxes, resolution=0.05):
    """ create the gt shape using input point clouds, bboxes, and ego transformations
    """
    shape_pc = np.zeros((0, 3))
    for i, (pc, bbox) in enumerate(zip(pcs, bboxes)):
        pc = sot_3d.utils.pc_in_box(bbox, pc, 1.0)
        bbox_state = BBox.bbox2array(bbox)[:4]
        pc -= bbox_state[:3]
        pc = sot_3d.utils.apply_motion_to_points(pc,
            np.array([0, 0, 0, -bbox.o]))
        
        shape_pc = np.vstack((shape_pc, pc))
        shape_pc = sot_3d.utils.downsample(shape_pc, voxel_size=resolution)

    return shape_pc


def main(bench_list_path, data_folder, token=0, process=1):
    gt_shape_folder = os.path.join(data_folder, 'gt_shapes')
    if not os.path.exists(gt_shape_folder):
        os.makedirs(gt_shape_folder)
    bench_list = json.load(open(bench_list_path, 'r'))
    
    for tracklet_index, tracklet_info in enumerate(bench_list):
        if tracklet_index % process != token:
            continue
        print('START {:} / {:}'.format(tracklet_index + 1, len(bench_list)))
        frame_range = tracklet_info['frame_range']
        segment_name = tracklet_info['segment_name']
        id = tracklet_info['id']
        pc_info = np.load(os.path.join(data_folder, 'pc', 'clean_pc', '{:}.npz'.format(segment_name)),
            allow_pickle=True)
        pcs = [pc_info[str(i)] for i in range(frame_range[0], frame_range[1] + 1)]
        bboxes = find_bboxes(id, data_folder, segment_name, frame_range[0], frame_range[1])
        shape_pc = create_gt_shape(pcs, bboxes)

        np.savez_compressed(os.path.join(gt_shape_folder, '{:}.npz'.format(id)),
            pc=shape_pc)


if __name__ == '__main__':
    if args.process > 1:
        pool = multiprocessing.Pool(args.process)
        for token in range(args.process):
            result = pool.apply_async(main, args=(args.bench_list, args.data_folder, token, args.process))
        pool.close()
        pool.join()
    else:
        main(args.bench_list, args.data_folder, 0, 1)