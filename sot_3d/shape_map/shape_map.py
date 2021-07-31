""" Maintaining a shape for the tracked vehicles
"""
import os, numpy as np, matplotlib.pyplot as plt
from copy import deepcopy
from ..shape_map.obj_shape import ObjShape
from ..shape_map.subshape import SubShape
from ..data_protos import BBox
from .. import utils, FrameData


class ShapeMap:
    def __init__(self, configs):
        # configuration
        self.id = id
        self.configs = configs
        self.window_size = self.configs['running']['window_size']
        self.update_freq = self.configs['shape_map']['update_freq']

        self.size = 0
        self.cur_frame = 0

        # data domain
        self.obj_shape = ObjShape(self.configs) # object shape logic
        self.shape_pc = None                    # point cloud of objects
        self.shape_state = None                 # where we place the shapes

        self.subshape = SubShape(self.configs)  # subshape logic
        self.subshape_pc = None                 # subshape point cloud
        self.subshape_state = None              # subshape state
    
    def initialize_shapes(self, pc, bbox: BBox):
        # after frame 0, no start_bbox and no initialization of shapes
        if not bbox:
            return
        self.obj_shape.add_pc(pc=pc, bbox=bbox)
    
    def pre_frame_optim(self, input_data: FrameData):
        # this initialization only works on frame 0
        self.initialize_shapes(pc=input_data.pc, bbox=input_data.start_bbox)
        self.shape_pc = self.obj_shape.shape_pc
        self.shape_state = self.obj_shape.shape_state

        self.size = min(self.size + 1, self.window_size)
        return

    def pre_optim_step(self, pred_states):
        result = list(dict() for _i in range(self.size))
        for _i in range(self.size - 1):
            result[_i]['shape'] = \
                ObjShape.trans2tgt_location(
                    self.shape_pc, self.shape_state, pred_states[_i])
            
            result[_i]['subshape'] = None
            if self.subshape_pc is not None:
                result[_i]['subshape'] = SubShape.trans2tgt_location(
                    self.subshape_pc, self.subshape_state, pred_states[_i])
        result[-1]['shape'] = None
        result[-1]['subshape'] = None
        return result
    
    def post_optim_step(self):
        return
    
    def post_frame_optim(self, pc, optim_bbox: BBox):
        """ update the appearance model
        """
        # for subshape, we update it after each frame
        self.subshape.add_pc(pc=pc, bbox=optim_bbox)
        self.subshape_pc = self.subshape.subshape_pc
        self.subshape_state = self.subshape.subshape_state
        
        # for object shape model, we update it according to certain intervals
        self.cur_frame += 1
        # frame 0 case
        if self.cur_frame - 1 == 0:
            return
        # no need for update case
        if (self.cur_frame - 1) % self.update_freq != 0:
            return
        # update the shape
        self.obj_shape.add_pc(pc=pc, bbox=optim_bbox)
        self.shape_pc = self.obj_shape.shape_pc
        self.shape_state = self.obj_shape.shape_state
        return
