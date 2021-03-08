""" This term regularize the consistency of motions
"""
import numpy as np
import sot_3d


class MotionConsistency:
    def __init__(self, configs):
        self.configs = configs
        self.prev_heading = None
    
    def pre_frame_optim(self, input_data: sot_3d.FrameData):
        return
    
    def pre_optim_step(self, optim_data: sot_3d.OptimData, frame_indexes):
        self.prev_heading = optim_data.bboxes[frame_indexes[0]].o
        return
    
    def post_optim_step(self):
        return
    
    def post_frame_optim(self):
        return
    
    def loss(self, params):
        # compute motion in intervals
        motion = np.reshape(params, (-1, 4))
        motion = np.sum(motion, axis=0)
        x, y, z, theta = motion

        v = np.linalg.norm([x, y]) + 1e-6
        dist = v * np.cos(self.prev_heading + theta / 2) - x
        return dist * dist
    
    def jac(self, params):
        # compute motion in intervals
        x, y, z, theta = params

        v = np.linalg.norm([x, y]) + 1e-6
        ref_heading = self.prev_heading + theta / 2
        cos_heading = np.cos(ref_heading)
        dist = v * cos_heading - x

        der_x = x * cos_heading / v - 1
        der_y = y * cos_heading / v
        der_z = 0
        der_o = -np.sin(ref_heading) * v / 2

        der = 2 * dist * np.array([der_x, der_y, der_z, der_o])
        return der 
