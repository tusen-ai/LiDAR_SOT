""" From frame 0 to frame 1, when no motion model is available, 
    find the initial location by registering the point cloud in a large region
"""
import numpy as np
from scipy.optimize import minimize, root
from .. import utils, factors, OptimData


class Finder:
    def __init__(self, configs, box, pc0, pc1):
        self.configs = configs
        self.box = box
        self.pc0 = pc0
        self.pc1 = pc1

        self.optim_method = configs['optim']['method']
        self.max_iter = configs['optim']['ITER']
        self.optim_options = configs['optim']['options']
    
    def icp(self):
        gt_bbox = self.box
        icp_loss = factors.ICPLoss(self.configs)

        ITER = 0
        motion = np.zeros(4)
        while ITER < self.max_iter:
            # prepare the optim data
            optim_data = OptimData()
            optim_data.pcs = [self.pc0, self.pc1]
            optim_data.bboxes = [gt_bbox, gt_bbox]
            optim_data.motions = [np.zeros(4), motion]

            icp_loss.pre_optim_step(optim_data, (0, 1))
            motion = np.zeros(4)
            res = minimize(icp_loss.loss, motion[:], method=self.optim_method,
                jac=icp_loss.jac, options=self.optim_options)
            
            motion = res.x
            ITER += 1
        print('************')
        return motion