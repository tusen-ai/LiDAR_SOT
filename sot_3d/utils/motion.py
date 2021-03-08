""" Tool functions manipulating motions 
"""
import numpy as np


all = ['reshape_motion',
       'agg_motion']


def reshape_motion(motions):
    if len(motions.shape) == 1:
        motions = motions.reshape((-1, 4))
    return motions


def agg_motion(optim_motions, frame_indexes):
    result = np.zeros(4)
    for _i in range(frame_indexes[0], frame_indexes[1] + 1):
        result += optim_motions[_i]
    return result