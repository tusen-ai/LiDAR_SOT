import sot_3d, numpy as np, os, argparse, json, multiprocessing
from sot_3d.data_protos import BBox
from sklearn.neighbors import NearestNeighbors
from sot_3d.visualization import o3d_pc_visualization


parser = argparse.ArgumentParser()
parser.add_argument('--name', default='debug', type=str, help='name of the experiment')
parser.add_argument('--result_folder', type=str, default='../TrackingResults/',
    help='path to store the tracking results')
parser.add_argument('--data_folder', type=str, default='../datasets/waymo/sot/',
    help='store the data')
parser.add_argument('--bench_list_folder', type=str, default='./benchmark', 
    help='the path of benchmark object list')
parser.add_argument('--process', type=int, default=1)
args = parser.parse_args()


HARDNESS = None


def str_list_to_int(lst):
    result = []
    for t in lst:
        try:
            t = int(t)
            result.append(t)
        except:
            continue
    return result


def neighbor_indices(pca, pcb):
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(pcb)
    _, indices = nbrs.kneighbors(pca)
    indices = indices.reshape(-1)
    return indices


def compute_cd_loss(pca, pcb):
    a2b_indices = neighbor_indices(pca, pcb)
    b2a_indices = neighbor_indices(pcb, pca)

    n_pca = np.concatenate((pca, pca[b2a_indices]), axis=0)
    n_pcb = np.concatenate((pcb[a2b_indices], pcb), axis=0)
    dist = n_pcb - n_pca
    dist = dist * dist
    dist = np.sqrt(np.sum(dist, axis=1))
    dist = np.sum(dist) / dist.shape[0]
    return dist


def create_shapes(pcs, bboxes, resolution=0.05):
    """ create the gt shape using input point clouds, bboxes, and ego transformations
    """
    assert len(pcs) == len(bboxes)
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


def pc2world(ego_matrix, pcs):
    new_pcs = np.concatenate((pcs,
                              np.ones(pcs.shape[0])[:, np.newaxis]),
                              axis=1)
    new_pcs = ego_matrix @ new_pcs.T
    new_pcs = new_pcs.T[:, :3]
    return new_pcs


def main(bench_list, result_folder, name, data_folder, token, process):
    pred_shape_folder = os.path.join(result_folder, name, 'shapes')
    results = list()
    for tracklet_index, tracklet_info in enumerate(bench_list):
        if tracklet_index % process != token:
            continue
        print('{:} {:} / {:}'.format(HARDNESS, tracklet_index + 1, len(bench_list)))
        id = tracklet_info['id']
        segment_name = tracklet_info['segment_name']
        frame_range = tracklet_info['frame_range']

        tracklet_results = json.load(open(
            os.path.join(result_folder, name, 'summary', '{:}.json'.format(id))))
        
        frame_keys = list(tracklet_results.keys())
        frame_keys = sorted(str_list_to_int(frame_keys))
        
        preds = list()
        for key in frame_keys:
            bbox = tracklet_results[str(key)]['bbox1']
            bbox = BBox.array2bbox(bbox)
            preds.append(bbox)
        preds.insert(0, BBox.array2bbox(tracklet_results[str(frame_keys[0])]['bbox0']))
        shape_file_path = os.path.join(pred_shape_folder, '{:}.npz'.format(id))
        if os.path.exists(shape_file_path):
            shape = np.load(shape_file_path, allow_pickle=True)['shape']
        else:
            scene_pcs = np.load(os.path.join(data_folder, 'pc', 'raw_pc', '{:}.npz'.format(segment_name)),
                allow_pickle=True)
            ego_infos = np.load(os.path.join(data_folder, 'ego_info', '{:}.npz'.format(segment_name)),
                allow_pickle=True)
            pcs = [pc2world(ego_infos[str(i)], scene_pcs[str(i)]) 
                for i in range(frame_range[0], frame_range[1] + 1)]
            shape = create_shapes(pcs, preds)
            np.savez_compressed(shape_file_path, shape=shape)

        gt_shape = np.load(os.path.join(data_folder, 'gt_shapes', '{:}.npz'.format(id)),
            allow_pickle=True)['pc']
        cd = compute_cd_loss(shape, gt_shape)
        results.append(cd)
    return results


if __name__ == '__main__':
    pred_shape_folder = os.path.join(args.result_folder, args.name, 'shapes')
    if not os.path.exists(pred_shape_folder):
        os.makedirs(pred_shape_folder)

    hardness_levels = ['easy', 'medium', 'hard']
    final_result = list()
    all_result = list()
    for hardness_level in hardness_levels:
        HARDNESS = hardness_level
        obj_list = json.load(open(os.path.join(args.bench_list_folder, '{:}.json'.format(hardness_level)), 'r'))
        
        pool = multiprocessing.Pool(args.process)
        results = pool.starmap(main, [
            (obj_list, args.result_folder, args.name, args.data_folder, token, args.process)
            for token in range(args.process)
        ])
        pool.close()
        pool.join()
        
        cds = list()
        for i in range(len(results)):
            for j in range(len(results[i])):
                cds.append(results[i][j])
        all_result += cds
        final_result.append(np.average(cds))
    
    for i, hardness_level in enumerate(hardness_levels):
        print('{:} \t: {:}'.format(hardness_level, final_result[i]))
    print('All Shape CDs {:}'.format(np.average(all_result)))