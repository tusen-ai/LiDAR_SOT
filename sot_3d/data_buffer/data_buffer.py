""" Managing the data for optimization, we use a buffer to store the related data
    1. point cloud (pc buffer)
    2. ego information (ego buffer)
    3. detection bboxes (det buffer, element is None on no detection cases)
"""
import os, numpy as np
from ..data_protos import BBox
from ..frame_data import FrameData
from .. import utils


class DataBuffer:
    def __init__(self, configs):
        # configuration
        self.configs = configs
        self.window_size = self.configs['running']['window_size']
        self.size = 0

        # data domain
        self.pc_buffer = utils.CircularBuffer(self.window_size, 0, 0)
        self.ego_buffer = utils.CircularBuffer(self.window_size, 0, 0)
        self.det_buffer = utils.CircularBuffer(self.window_size, 0, 0)
        self.terrain = None
    
    def pre_frame_optim(self, input_data: FrameData):
        """ on receiving each frame, store their point cloud
        Args:
            input_data (FrameData): input data
        """
        self.pc_buffer.push(input_data.pc)
        self.ego_buffer.push(input_data.ego)
        self.det_buffer.push(input_data.dets)
        self.size = self.pc_buffer.size
        self.terrain = input_data.terrain
        return
    
    def pre_optim_step(self):
        """ Before each optimization step, the motion model prepare the following data
            1. pc: raw point cloud of each frame
            2. ego information: ego matrix
            data buffer return a list of dict. [{'pc': xxx, 'ego': xxx}, ..., ...]
            Each item corresponds to the data of a frame
        """
        result = [dict() for i in range(self.size)]
        for i in range(self.size):
            result[i]['pc'] = self.pc_buffer.access(i)
            result[i]['ego'] = self.ego_buffer.access(i)
            result[i]['dets'] = self.det_buffer.access(i)
            result[i]['terrain'] = self.terrain
        return result
    
    def post_optim_step(self):
        """ data buffer has nothing to do on each post optimization step
        """
        return

    def post_frame_optim(self):
        """ data buffer has nothing to do on each post optimization step
        """
        return
