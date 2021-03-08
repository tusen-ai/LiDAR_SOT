import numpy as np
from copy import deepcopy
import sot_3d
import sot_3d.utils as utils
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sot_3d.data_protos import BBox
from sot_3d.visualization import Visualizer2D


class ShapeLoss:
    def __init__(self, configs):
        self.shape_pc = None         # shape point clouds
        self.shape_state = None      # shape state
        self.pc = None               # current frame pc
        self.state = None            # current frame pred box
        self.diff_state = None
        
        self.configs = configs
        self.neighbor_num = configs['neighbor_num']
        self.least_pc = configs['least_pc']
        self.box_scaling_next = self.configs['box_scaling_next']
        self.pc_limit = configs['pc_limit']
        self.loss_type = configs['loss_type']
        self.huber_limit = 0.5       # if use huber loss, the threshold for huber loss

        self.ransac = configs['ransac']
        self.ransac_on = self.ransac['switch']
        if self.ransac_on:
            self.ransac_num_iter = self.ransac['num_iter']
            self.ransac_threshold = self.ransac['threshold']
            self.ransac_limit = self.ransac['ransac_limit']
        return
    
    def pre_frame_optim(self, input_data: sot_3d.FrameData):
        return
    
    def pre_optim_step(self, optim_data: sot_3d.OptimData, frame_indexes):
        prev_bbox = optim_data.bboxes[frame_indexes[0]] 
        pred_bbox = optim_data.bboxes[frame_indexes[1]]
        self.shape_state = BBox.bbox2array(prev_bbox)[:4]
        self.state = BBox.bbox2array(pred_bbox)[:4]

        self.shape_pc = optim_data.shape_pcs[frame_indexes[0]]
        self.pc = optim_data.pcs[frame_indexes[1]]
        self.pc = utils.pc_in_box(pred_bbox, self.pc, self.box_scaling_next)

        self.shape_pc -= self.shape_state[:3]
        self.pc -= self.shape_state[:3] 

        reference_motion = self.state - self.shape_state
        self.shape_pc, self.pc = self.find_nearest_neighbor(
            self.shape_pc, self.pc, reference_motion)
    
    def post_optim_step(self):
        return
    
    def post_frame_step(self):
        return
    
    def find_nearest_neighbor(self, pc_a, pc_b, motion):
        """ find the nearest neighbor pairs given the point clouds and their estimated motions
        Args:
            motion: an estimated motion
        Return:
            pc_a, pc_b: corresponding point pairs
        """
        if pc_a is None or pc_b is None:
            return None, None
        if pc_a.shape[0] < self.least_pc or pc_b.shape[0] < self.least_pc:
            return None, None 

        pc_a, pc_b = deepcopy(pc_a), deepcopy(pc_b)
        # subsample when the point number is too large
        if pc_a.shape[0] > self.pc_limit:
            indices = (pc_a.shape[0] * np.random.uniform(size=self.pc_limit)).astype(np.int)
            pc_a = pc_a[indices]
        if pc_b.shape[0] > self.pc_limit:
            indices = (pc_b.shape[0] * np.random.uniform(size=self.pc_limit)).astype(np.int)
            pc_b = pc_b[indices]

        moved_pc_a = utils.apply_motion_to_points(pc_a, motion)
        nbrs = NearestNeighbors(n_neighbors=self.neighbor_num, algorithm='kd_tree').fit(pc_b)
        _, indices = nbrs.kneighbors(moved_pc_a)
        indices = indices.reshape(-1)
        pc_b = pc_b[indices]

        if self.ransac_on:
            mask = utils.ransac(
                src=pc_a, tgt=pc_b, min_point_num=3, num_iter=self.ransac_num_iter,
                threshold=self.ransac_threshold, reference_motion=motion)
            if np.sum(mask) >= self.ransac_limit:
                pc_a, pc_b = pc_a[mask], pc_b[mask]

        return pc_a, pc_b

    def loss(self, params):
        if self.shape_pc is None or self.pc is None:
            return 0
        # relative location by subtracting starting location
        relative_motion = params
        # update to new locations
        point_num = self.shape_pc.shape[0]
        x, y, z, theta = relative_motion
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta),  np.cos(theta), 0],
                                    [0            ,  0            , 1]])
        new_pcs = rotation_matrix @ (self.shape_pc).T
        new_pcs = new_pcs.T[:, :3] + np.asarray([x, y, z])

        # compute the loss
        dist = self.pc - new_pcs
        dist = dist * dist
        loss = np.sum(dist) / point_num

        return loss
    
    def jac(self, params):
        if self.shape_pc is None or self.pc is None:
            return np.zeros(params.shape[0])
        # relative location by subtracting starting location
        relative_motion = params
        # update to new locations
        x, y, z, theta = relative_motion
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta),  np.cos(theta), 0],
                                    [0            ,  0            , 1]])
        new_pcs = rotation_matrix @ (self.shape_pc).T
        new_pcs = new_pcs.T[:, :3] + np.asarray([x, y, z])
        dist = self.pc - new_pcs

        # Derivative of x, y
        derive = -2 * np.average(dist[:, :3], axis=0)
        derive_x, derive_y, derive_z = derive

        # Derivative of theta
        tmp_transformation = np.array([[ np.sin(theta), -np.cos(theta)], 
                                       [ np.cos(theta),  np.sin(theta)]])
        derive_theta_pcs = self.shape_pc[:, :2] @ tmp_transformation
        # here we get [N * [xsin + ycos, -xcos + ysin]]
        derive_theta = np.sum(dist[:, :2] * derive_theta_pcs, axis=1)
        derive_theta = np.average(derive_theta)
        derive_theta *= 2

        der = np.asarray([derive_x, derive_y, derive_z, derive_theta])
        return der
