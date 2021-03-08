""" Get the heigh of ground from a terrain map
"""
import numpy as np
from sot_3d.utils.geometry import pc_in_box_2D


__all__ = ['get_latitude']


def get_latitude(box, ground_map, scaling=1.0):
    """ query the latitude in the area specified by box 
    """
    ground = pc_in_box_2D(box, ground_map, scaling)
    z = np.average(ground, axis=0)[2]
    return z