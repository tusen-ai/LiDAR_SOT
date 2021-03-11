""" rectify the bbox with size and corner  
"""
import numpy as np, math
from copy import deepcopy
from sot_3d.data_protos import BBox


__all__ = [
    'bbox_rectify'
]


def bbox_rectify(bbox: BBox, size, ego):
    """ rectify the bbox using its corner
    Args:
        bbox (BBox): input bbox
        size: [length, width, height]
        ego: [x, y, z]
    """
    # 1. determine the viewable point of the box
    view_index, view_location = viewable_corner(bbox, ego)
    # 2. extend the box
    prev_location = view_location
    corners = np.zeros((4, 2))
    corners[view_index, :] = view_location
    for _i in range(view_index + 1, view_index + 4):
        end_index = _i % 4
        
        # how long is this edge
        if end_index % 2 == 0:
            new_length = size[0]
        else:
            new_length = size[1]
        
        comp_heading = bbox.o + (np.pi / 2) * end_index
        x = prev_location[0] + new_length * np.cos(comp_heading)
        y = prev_location[1] + new_length * np.sin(comp_heading)
        corners[end_index, :] = np.array([x, y])
        prev_location = np.array([x, y])
    # 3. reformat
    center = (corners[0] + corners[2]) / 2
    result = deepcopy(bbox)
    result.x = center[0]
    result.y = center[1]
    result.l = size[0]
    result.w = size[1]

    # 4. move the center z
    result.h = size[2]
    result.z = bbox.z - bbox.h / 2 + result.h / 2

    return result



def viewable_corner(bbox: BBox, ego):
    # nearest corner, viewable corner
    ego_location = ego[:2]
    corners = np.array(BBox.box2corners2d(bbox))[:, :2]

    dist = corners - ego_location
    dist = np.sum(dist * dist, axis=1)
    corner_index = np.argmin(dist)
    corner_coordiante = corners[corner_index]
    return corner_index, corner_coordiante