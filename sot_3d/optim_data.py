""" The data format containing all the information for optimization
    In the pre_optim_step part, every factor fetches its desired data from this
    The structure of optim_data is:
    1. Each attribute corresponds to a type of data
    2. Each item in the list corresponds to a frame
"""
import numpy as np
from . import utils


class OptimData:
    def __init__(self, size=0):
        self.size = size           # how many frames to consider simultaneously
        self.pcs = list()          # point cloud
        self.bboxes = list()       # pred bboxes
        self.motions = list()      # reference motions
        self.egos = list()         # ego matrix
        self.dets = list()         # detection input, if any
        
        self.shape_pcs = list()    # shape of objects
        self.subshape_pcs = list() # subshapes of objects
    
    @classmethod
    def dict2optim_data(cls, raw_optim_data):
        """ Transforming a dict from pre_optim_step to optim_data
        """
        result = OptimData()
        result.size = len(raw_optim_data)

        result.pcs = [raw_optim_data[i]['pc'] for i in range(result.size)]
        result.bboxes = [raw_optim_data[i]['bbox'] for i in range(result.size)]
        result.motions = [raw_optim_data[i]['motion'] for i in range(result.size)]
        result.egos = [raw_optim_data[i]['ego'] for i in range(result.size)]
        result.dets = [raw_optim_data[i]['dets'] for i in range(result.size)]
        result.shape_pcs = [raw_optim_data[i]['shape'] for i in range(result.size)]
        result.subshape_pcs = [raw_optim_data[i]['subshape'] for i in range(result.size)]
        result.terrain = raw_optim_data[0]['terrain']
        return result
        