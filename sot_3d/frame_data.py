""" The format for input data for the tracker
    The following things included:
    1. Compulsory: ego_info, point cloud
    2. Optional: terrain information, detections, etc.
    3. You may add your own field of data into frame data
"""
import numpy as np
import sot_3d.utils as utils
from sot_3d.data_protos import BBox
from sot_3d.visualization import Visualizer2D


class FrameData:
    def __init__(self, ego_info, pc, start_bbox=None, terrain=None, dets=None):
        """ the input of a frame should contain several fields, some compulsory, some optional
        Args:
            ego_info (4 * 4 matrix): ego matrix
            pc (N * 3 array): point cloud
            terrain (N * 3 pc terrain map, optional): ground point cloud. Defaults to None.
            dets (BBoxes, optional): detection bboxes. Defaults to None.
        """
        self.ego = ego_info
        self.pc = pc
        self.terrain = terrain
        self.dets = None
        if dets is not None:
            self.dets = [BBox.array2bbox(det) for det in dets]
        self.start_bbox = start_bbox
        self.preprocess()
        return
    
    def preprocess(self):
        """ Some data need eplicit transformation to the world coordinate:
            point cloud, detection bboxes
        """
        self.pc = utils.pc2world(self.ego, self.pc)
        if self.dets is not None:
            self.dets = [BBox.bbox2world(self.ego, det) for det in self.dets]
        return
