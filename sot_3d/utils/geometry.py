import numpy as np
from copy import deepcopy
from shapely.geometry import Polygon
import numba
from sot_3d.data_protos import BBox


__all__ = ['pc_in_box', 'downsample', 'pc_in_box_2D',
           'apply_motion_to_points', 'make_transformation_matrix',
           'iou2d', 'iou3d', 'pc_in_box_range', 'pc_in_box_range_2D',
           'pc2world']


def apply_motion_to_points(points, motion, pre_move=0):
    transformation_matrix = make_transformation_matrix(motion)
    points = deepcopy(points)
    points = points + pre_move
    new_points = np.concatenate((points,
                                 np.ones(points.shape[0])[:, np.newaxis]),
                                 axis=1)

    new_points = transformation_matrix @ new_points.T
    new_points = new_points.T[:, :3]
    new_points -= pre_move
    return new_points


@numba.njit
def downsample(points, voxel_size=0.05):
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


# def pc_in_box(box, pc, box_scaling=1.5):
#     center_x, center_y, length, width = \
#         box['center_x'], box['center_y'], box['length'], box['width']
#     center_z, height = box['center_z'], box['height']
#     yaw = box['heading']

#     rx = np.abs((pc[:, 0] - center_x) * np.cos(yaw) + (pc[:, 1] - center_y) * np.sin(yaw))
#     ry = np.abs((pc[:, 0] - center_x) * -(np.sin(yaw)) + (pc[:, 1] - center_y) * np.cos(yaw))
#     rz = np.abs(pc[:, 2] - center_z)

#     mask_x = (rx < (length * box_scaling / 2))
#     mask_y = (ry < (width * box_scaling / 2))
#     mask_z = (rz < (height / 2))

#     mask = mask_x * mask_y * mask_z
#     indices = np.argwhere(mask == 1).reshape(-1)
#     return pc[indices, :]


# def pc_in_box_2D(box, pc, box_scaling=1.0):
#     center_x, center_y, length, width = \
#         box['center_x'], box['center_y'], box['length'], box['width']
#     yaw = box['heading']
    
#     cos = np.cos(yaw) 
#     sin = np.sin(yaw)
#     rx = np.abs((pc[:, 0] - center_x) * cos + (pc[:, 1] - center_y) * sin)
#     ry = np.abs((pc[:, 0] - center_x) * -(sin) + (pc[:, 1] - center_y) * cos)

#     mask_x = (rx < (length * box_scaling / 2))
#     mask_y = (ry < (width * box_scaling / 2))

#     mask = mask_x * mask_y
#     indices = np.argwhere(mask == 1).reshape(-1)
#     return pc[indices, :]


def pc_in_box(box, pc, box_scaling=1.5):
    center_x, center_y, length, width = \
        box.x, box.y, box.l, box.w
    center_z, height = box.z, box.h
    yaw = box.o
    return pc_in_box_inner(center_x, center_y, center_z, length, width, height, yaw, pc, box_scaling)


@numba.njit
def pc_in_box_inner(center_x, center_y, center_z, length, width, height, yaw, pc, box_scaling=1.5):
    mask = np.zeros(pc.shape[0], dtype=np.int32)
    yaw_cos, yaw_sin = np.cos(yaw), np.sin(yaw)
    for i in range(pc.shape[0]):
        rx = np.abs((pc[i, 0] - center_x) * yaw_cos + (pc[i, 1] - center_y) * yaw_sin)
        ry = np.abs((pc[i, 0] - center_x) * -yaw_sin + (pc[i, 1] - center_y) * yaw_cos)
        rz = np.abs(pc[i, 2] - center_z)

        if rx < (length * box_scaling / 2) and ry < (width * box_scaling / 2) and rz < (height * box_scaling / 2):
            mask[i] = 1
    indices = np.argwhere(mask == 1)
    result = np.zeros((indices.shape[0], 3), dtype=np.float64)
    for i in range(indices.shape[0]):
        result[i, :] = pc[indices[i], :]
    return result


def pc_in_box_2D(box, pc, box_scaling=1.0):
    center_x, center_y, length, width = \
        box.x, box.y, box.l, box.w
    center_z, height = box.z, box.h
    yaw = box.o
    return pc_in_box_2D_inner(center_x, center_y, center_z, length, width, height, yaw, pc, box_scaling)


@numba.njit
def pc_in_box_2D_inner(center_x, center_y, center_z, length, width, height, yaw, pc, box_scaling=1.0):
    mask = np.zeros(pc.shape[0], dtype=np.int32)
    yaw_cos, yaw_sin = np.cos(yaw), np.sin(yaw)
    for i in range(pc.shape[0]):
        rx = np.abs((pc[i, 0] - center_x) * yaw_cos + (pc[i, 1] - center_y) * yaw_sin)
        ry = np.abs((pc[i, 0] - center_x) * -yaw_sin + (pc[i, 1] - center_y) * yaw_cos)

        if rx < (length * box_scaling / 2) and ry < (width * box_scaling / 2):
            mask[i] = 1
    indices = np.argwhere(mask == 1)
    result = np.zeros((indices.shape[0], 3), dtype=np.float64)
    for i in range(indices.shape[0]):
        result[i, :] = pc[indices[i], :]
    return result


def make_transformation_matrix(motion):
    x, y, z, theta = motion
    transformation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0, x],
                                      [np.sin(theta),  np.cos(theta), 0, y],
                                      [0            ,  0            , 1, z],
                                      [0            ,  0            , 0, 1]])
    return transformation_matrix


def iou2d(box_a, box_b):
    boxa_corners = np.array(BBox.box2corners2d(box_a))[:, :2]
    boxb_corners = np.array(BBox.box2corners2d(box_b))[:, :2]
    reca, recb = Polygon(boxa_corners), Polygon(boxb_corners)
    overlap = reca.intersection(recb).area
    area_a = reca.area
    area_b = recb.area
    iou = overlap / (area_a + area_b - overlap + 1e-10)
    return iou


def iou3d(box_a, box_b):
    boxa_corners = np.array(BBox.box2corners2d(box_a))
    boxb_corners = np.array(BBox.box2corners2d(box_b))[:, :2]
    reca, recb = Polygon(boxa_corners), Polygon(boxb_corners)
    overlap_area = reca.intersection(recb).area
    iou_2d = overlap_area / (reca.area + recb.area - overlap_area)
    ha, hb = box_a.h, box_b.h
    za, zb = box_a.z, box_b.z
    overlap_height = max(0, min((za + ha / 2) - (zb - hb / 2), (zb + hb / 2) - (za - ha / 2)))
    overlap_volume = overlap_area * overlap_height
    union_volume = box_a.w * box_a.l * ha + box_b.w * box_b.l * hb - overlap_volume
    iou_3d = overlap_volume / union_volume

    union_height = max(za + ha / 2, zb + hb / 2) - min(za - ha / 2, zb - hb / 2)
    return iou_2d, iou_3d


def pc_in_box_range(box, pc, dist=5.0):
    center_x, center_y, length, width = \
        box['center_x'], box['center_y'], box['length'], box['width']
    center_z, height = box['center_z'], box['height']
    yaw = box['heading']

    result = pc_in_box_range_inner(center_x, center_y, center_z, pc, dist)
    return result


@numba.njit
def pc_in_box_range_inner(center_x, center_y, center_z, pc, dist):
    mask = np.zeros(pc.shape[0], dtype=np.int32)
    for i in range(pc.shape[0]):
        rx = np.abs(pc[i, 0] - center_x)
        ry = np.abs(pc[i, 0] - center_x)
        rz = np.abs(pc[i, 2] - center_z)

        if rx < dist and ry < dist and rz < dist:
            mask[i] = 1
    indices = np.argwhere(mask == 1)
    result = np.zeros((indices.shape[0], 3), dtype=np.float64)
    for i in range(indices.shape[0]):
        result[i, :] = pc[indices[i], :]
    return result


def pc_in_box_range_2D(box, pc, dist=5.0):
    center_x, center_y, length, width = \
        box['center_x'], box['center_y'], box['length'], box['width']
    center_z, height = box['center_z'], box['height']
    yaw = box['heading']

    result = pc_in_box_range_2D_inner(center_x, center_y, center_z, pc, dist)
    return result


@numba.njit
def pc_in_box_range_2D_inner(center_x, center_y, center_z, pc, dist):
    mask = np.zeros(pc.shape[0], dtype=np.int32)
    for i in range(pc.shape[0]):
        rx = np.abs(pc[i, 0] - center_x)
        ry = np.abs(pc[i, 0] - center_x)

        if rx < dist and ry < dist:
            mask[i] = 1
    indices = np.argwhere(mask == 1)
    result = np.zeros((indices.shape[0], 3), dtype=np.float64)
    for i in range(indices.shape[0]):
        result[i, :] = pc[indices[i], :]
    return result


def pc2world(ego_matrix, pcs):
    new_pcs = np.concatenate((pcs,
                              np.ones(pcs.shape[0])[:, np.newaxis]),
                              axis=1)
    new_pcs = ego_matrix @ new_pcs.T
    new_pcs = new_pcs.T[:, :3]
    return new_pcs