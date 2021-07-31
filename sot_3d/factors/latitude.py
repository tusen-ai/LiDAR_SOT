""" Loss term regularizing the latitude of objects.
    Its intuition is to put the objects on the ground.
    The formula is to compute the L2 distance between current frame latitude with a reference latitude (frame 0 latitude by default).
"""
import numpy as np, math
from .. import utils,FrameData, OptimData


class LatitudeLoss:
    def __init__(self, configs):
        self.terrain_scaling = configs['scaling'] # parameter for querying the ground height
        self.ref_latitude_set = False             # set the reference latitude already?
        self.ref_latitude = None                  # reference latitude
        self.prev_latitude = None                 # latitude in the previous frame
        self.cur_ground_latitude = None           # ground latitude in the current frame
        return
    
    def pre_frame_optim(self, input_data: FrameData):
        """ set the reference latitude on frame 0
        """
        if self.ref_latitude_set:
            return
        self.ref_latitude_set = True
        terrain = input_data.terrain
        bbox = input_data.start_bbox
        latitude = utils.get_latitude(bbox, terrain, self.terrain_scaling)

        if math.isnan(latitude):
            self.ref_latitude = None
        else:
            self.ref_latitude = bbox.z - latitude
        return
    
    def pre_optim_step(self, optim_data: OptimData, frame_indexes):
        """ set the newest previous latitude, so that we can compute the loss w.r.t motion
        """
        self.prev_latitude = optim_data.bboxes[frame_indexes[0]].z
        
        cur_bbox = optim_data.bboxes[frame_indexes[1]]
        terrain = optim_data.terrain
        self.cur_ground_latitude = utils.get_latitude(cur_bbox, terrain, self.terrain_scaling)
        if math.isnan(self.cur_ground_latitude):
            self.cur_ground_latitude = None
    
    def post_optim_step(self):
        return
    
    def post_frame_optim(self):
        return
    
    def loss(self, params):
        if not (self.ref_latitude and self.prev_latitude and self.cur_ground_latitude):
            return 0
        
        dist = params[2] + self.prev_latitude - self.cur_ground_latitude - self.ref_latitude
        return dist * dist
    
    def jac(self, params):
        if not (self.ref_latitude and self.prev_latitude and self.cur_ground_latitude):
            return np.zeros(4)
        
        dist = params[2] + self.prev_latitude - self.cur_ground_latitude - self.ref_latitude
        derivative = np.asarray([0, 0, 2 * dist, 0])
        return derivative
