""" Overlay the ground point cloud on each frame together as a pseudo terrain map
"""
import numpy as np, os, argparse, numba, multiprocessing


parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', type=str, default='../../../datasets/waymo/sot',
    help='the location of data')
parser.add_argument('--process', type=int, default=1, help='multiprocessing for acceleration')
args = parser.parse_args()
args.ground_pc_folder = os.path.join(args.data_folder, 'pc', 'ground_pc')
args.terrain_map_folder = os.path.join(args.data_folder, 'pc', 'terrain_pc')
args.ego_info_folder = os.path.join(args.data_folder, 'ego_info')


@numba.njit
def downsample(points, voxel_size=0.10):
    sample_dict = dict()
    for i in range(points.shape[0]):
        point_coord = np.floor(points[i] / voxel_size)
        sample_dict[(int(point_coord[0]), int(point_coord[1]), int(point_coord[2]))] = True
    res = np.zeros((len(sample_dict), 3), dtype=np.float32)
    idx = 0
    for k, v in sample_dict.items():
        res[idx, 0] = k[0] * voxel_size + voxel_size / 2
        res[idx, 1] = k[1] * voxel_size + voxel_size / 2
        res[idx, 2] = k[2] * voxel_size + voxel_size / 2
        idx += 1
    return res


def pc2world(ego_matrix, pcs):
    new_pcs = np.concatenate((pcs,
                              np.ones(pcs.shape[0])[:, np.newaxis]),
                              axis=1)
    new_pcs = ego_matrix @ new_pcs.T
    new_pcs = new_pcs.T[:, :3]
    return new_pcs


def main(ego_info_folder, ground_pc_folder, terrain_map_folder, token=0, process=1):
    file_names = sorted(os.listdir(ground_pc_folder))
    for file_index, file_name in enumerate(file_names):
        if file_index % process != token:
            continue
        print('START SEQ {:} / {:}'.format(file_index + 1, len(file_names)))
        ground_pc_data = np.load(os.path.join(ground_pc_folder, file_name), allow_pickle=True)
        ego_info_data = np.load(os.path.join(ego_info_folder, file_name), allow_pickle=True)
        frame_keys = list(ground_pc_data.keys())

        terrain = np.empty((0, 3))
        for i, frame_key in enumerate(frame_keys):
            pc = ground_pc_data[frame_key]
            ego_matrix = ego_info_data[frame_key]
            pc = pc2world(ego_matrix, pc)
            terrain = np.vstack((terrain, pc))
            terrain = downsample(terrain)
            if (i + 1) % 10 == 0:
                print('SEQ {:} / {:} Frame {:} / {:}'.format(file_index + 1, len(file_names), i + 1, len(frame_keys)))
        
        np.savez_compressed(os.path.join(terrain_map_folder, file_name), terrain=terrain)


if __name__ == '__main__':
    pool = multiprocessing.Pool(args.process)
    for token in range(args.process):
        result = pool.apply_async(main, args=(args.ego_info_folder, args.ground_pc_folder, args.terrain_map_folder, token, args.process))
    pool.close()
    pool.join()