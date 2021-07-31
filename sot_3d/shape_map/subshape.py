import numpy as np
from .. import utils
from ..data_protos import BBox


class SubShape:
    def __init__(self, configs):
        self.subshape_len = configs['shape_map']['subshape_len']
        self.box_scaling = configs['shape_map']['box_scaling_pc_bank']
        self.downsample = configs['shape_map']['downsample']
        self.resolution = configs['shape_map']['resolution']

        self.pc_buffer = utils.CircularBuffer(self.subshape_len, 0, 0)
        self.state_buffer = utils.CircularBuffer(self.subshape_len, 0, 0)
        self.size = 0

        self.subshape_pc = None
        self.subshape_state = None
    
    def add_pc(self, pc, bbox: BBox):
        """ find the current frame pc, append it to buffer, then update the subshape
        """
        cur_frame_pc = utils.pc_in_box(bbox, pc, self.box_scaling)
        bbox_state = BBox.bbox2array(bbox)[:4]
        if self.downsample:
            cur_frame_pc = utils.downsample(cur_frame_pc, self.resolution)
        
        self.pc_buffer.push(cur_frame_pc)
        self.state_buffer.push(bbox_state)
        self.size = self.pc_buffer.size

        self.update_subshape()
        return
    
    def update_subshape(self):
        if self.size == 0:
            return
        
        self.subshape_pc = self.pc_buffer.access(0)
        self.subshape_state = self.state_buffer.access(0)
        for i in range(1, self.size):
            pc = self.pc_buffer.access(i)
            state = self.state_buffer.access(i)
            pc = SubShape.trans2tgt_location(pc, state, self.subshape_state)
            self.subshape_pc = np.vstack((self.subshape_pc, pc))
        
        if self.downsample:
            self.subshape_pc = utils.downsample(self.subshape_pc, self.resolution)
    
    @classmethod
    def trans2tgt_location(cls, pc, src_state, tgt_state):
        pc = pc - src_state[:3]
        pc = utils.apply_motion_to_points(pc, 
            np.array([0, 0, 0, tgt_state[3] - src_state[3]]))
        pc += tgt_state[:3]
        return pc