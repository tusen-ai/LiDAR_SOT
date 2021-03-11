import numpy as np


__all__ = [
    'str2int',
    'box2corners2d',
    'pc2world',
    'location2yaw',
    'bbox2world'
]


def str2int(strs):
    result = [int(s) for s in strs]
    return result


def box2corners2d(box):
    center_x, center_y, length, width = \
        box[0], box[1], box[4], box[5]
    yaw = box[3]
    center_z, height = box[2], box[6]

    center_point = np.array([center_x, center_y, center_z])
    bottom_center = np.array([center_x, center_y, center_z - height / 2])
    pc0 = np.array([center_x + np.cos(yaw) * length / 2 + np.sin(yaw) * width / 2,
                    center_y + np.sin(yaw) * length / 2 - np.cos(yaw) * width / 2,
                    center_z - height / 2])
    pc1 = np.array([center_x + np.cos(yaw) * length / 2 - np.sin(yaw) * width / 2,
                    center_y + np.sin(yaw) * length / 2 + np.cos(yaw) * width / 2,
                    center_z - height / 2])
    pc2 = 2 * bottom_center - pc0
    pc3 = 2 * bottom_center - pc1

    return [pc0.tolist(), pc1.tolist(), pc2.tolist(), pc3.tolist()]


def pc2world(ego_matrix, pcs):
    new_pcs = np.concatenate((pcs,
                              np.ones(pcs.shape[0])[:, np.newaxis]),
                              axis=1)
    new_pcs = ego_matrix @ new_pcs.T
    new_pcs = new_pcs.T[:, :3]
    return new_pcs


def location2yaw(center, edge):
    vec = edge - center
    yaw = np.arccos(vec[0] / np.linalg.norm(vec))
    if vec[1] < 0:
        yaw = -yaw
    return yaw


def bbox2world(ego_matrix, box):
    # center and corners
    corners = np.array(box2corners2d(box))
    center = box[:3][np.newaxis, :]
    center = pc2world(ego_matrix, center)[0]
    corners = pc2world(ego_matrix, corners)
    # heading
    edge_mid_point = (corners[0] + corners[1]) / 2
    yaw = location2yaw(center[:2], edge_mid_point[:2])
    
    result = np.zeros(7)
    result[:3] = center
    result[3] = yaw
    result[4:] = box[4:]
    return result