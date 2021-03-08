import numpy as np
import sot_3d.utils as utils
from sot_3d.data_protos import BBox
import sot_3d


class ObjShape:
    def __init__(self, configs=None, frame_key=None):
        self.window_size = configs['running']['window_size']
        self.box_scaling = configs['shape_map']['box_scaling_pc_bank']
        self.downsample = configs['shape_map']['downsample']
        self.resolution = configs['shape_map']['resolution']

        self.shape_pc = None      # shape point cloud
        self.shape_state = None   # where do we place the shapes
    
    def add_pc(self, pc, bbox: BBox):
        template_pc = utils.pc_in_box(bbox, pc, self.box_scaling)
        bbox_state = BBox.bbox2array(bbox)[:4]
        if self.shape_pc is None:
            self.shape_pc = template_pc
            self.shape_state = bbox_state
        else:
            new_pc = ObjShape.trans2tgt_location(template_pc, bbox_state, self.shape_state)
            self.shape_pc = np.vstack((self.shape_pc, new_pc))
        
        if self.downsample:
            self.shape_pc = utils.downsample(self.shape_pc, voxel_size=self.resolution)
    
    @classmethod
    def trans2tgt_location(cls, pc, src_state, tgt_state):
        pc = pc - src_state[:3]
        pc = utils.apply_motion_to_points(pc, 
            np.array([0, 0, 0, tgt_state[3] - src_state[3]]))
        pc += tgt_state[:3]
        return pc
