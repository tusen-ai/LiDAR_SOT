import numpy as np, math
from .. import utils, OptimData, FrameData


class MotionPriorFactor:
    """ near to the result from previous iteration or motion model prediction
    """
    def __init__(self, configs):
        self.reference_motions = None
        return
    
    def pre_frame_optim(self, input_data: FrameData):
        return
    
    def pre_optim_step(self, optim_data: OptimData, frame_indexes):
        frame_start, frame_end = frame_indexes
        self.reference_motions = np.zeros(4 * (frame_end - frame_start))
        for _i in range(frame_start + 1, frame_end + 1):
            self.reference_motions[4 * (_i - 1): 4 * _i] = \
                optim_data.motions[_i]
        return
    
    def post_optim_step(self):
        self.reference_motions = None
        return
    
    def post_frame_optim(self):
        return
    
    def loss(self, params):
        dist = params - self.reference_motions
        dist = np.sum(dist * dist)
        return dist
    
    def jac(self, params):
        dist = params - self.reference_motions
        derivative = 2 * dist
        return derivative
