""" Point-to-Point Loss between LiDAR scans
"""
import numpy as np, numba, sot_3d
from copy import deepcopy
import sot_3d.utils as utils
from sklearn.neighbors import NearestNeighbors
from sot_3d.data_protos import BBox


class ICPLoss:
    def __init__(self, configs):
        self.pc_a = None
        self.pc_b = None

        self.configs = configs
        self.neighbor_num = configs['neighbor_num']
        self.least_pc = configs['least_pc']
        self.pc_limit = configs['pc_limit']
        self.loss_type = configs['loss_type']
        self.huber_limit = 0.5 # if use huber loss, the threshold for huber loss

        self.box_scaling_prev = self.configs['box_scaling_prev']
        self.box_scaling_next = self.configs['box_scaling_next']
        self.agg_subshape = self.configs['agg_subshape']
    
    def pre_frame_optim(self, input_data: sot_3d.FrameData):
        return
    
    def pre_optim_step(self, optim_data: sot_3d.OptimData, frame_indexes):
        """ prepare the loss computation for icp loss
        Args:
            optim_data (sot_3d.OptimData): data for optimization
            frame_indexes ((frame0, frame1)): which two frames to compute
        """
        self.pc_a = optim_data.pcs[frame_indexes[0]]
        self.pc_b = optim_data.pcs[frame_indexes[1]]

        box_a = optim_data.bboxes[frame_indexes[0]]
        box_b = optim_data.bboxes[frame_indexes[1]]

        # prepare relevant LiDAR points
        self.pc_a = utils.pc_in_box(box_a, self.pc_a, self.box_scaling_prev)
        self.pc_b = utils.pc_in_box(box_b, self.pc_b, self.box_scaling_next)

        if self.agg_subshape:
            subshape_pc = optim_data.subshape_pcs[frame_indexes[0]]
            if subshape_pc is not None:
                self.pc_a = np.vstack((self.pc_a, subshape_pc))
        
        # subtracting the center of objects
        # so that we fit the computation of rigid motion
        point_center = BBox.bbox2array(box_a)[:3]
        self.pc_a -= point_center[:3]
        self.pc_b -= point_center[:3]
        # find correspondences
        reference_motion = utils.agg_motion(optim_data.motions, 
            (frame_indexes[0] + 1, frame_indexes[1]))
        self.pc_a, self.pc_b = self.find_nearest_neighbor(self.pc_a, self.pc_b, reference_motion)
        # set frame interval
        self.frame_range = (frame_indexes[0], frame_indexes[1])
    
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
        return pc_a, pc_b
    
    def post_optim_step(self):
        return
    
    def post_frame_optim(self):
        return
    
    def loss(self, params):
        """ Compute the icp loss in the self.frame_range specified interval
        """
        if self.pc_a is None or self.pc_b is None:
            return 0
        # compute motion in intervals
        motion = np.reshape(params, (-1, 4))
        motion = np.sum(motion, axis=0)

        # update to new locations
        point_num = self.pc_a.shape[0]
        x, y, z, theta = motion
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta),  np.cos(theta), 0],
                                    [0,              0,             1]])
        new_pcs = rotation_matrix @ (self.pc_a).T
        new_pcs = new_pcs.T[:, :3]
        new_pcs += np.array([x, y, z])

        # compute the loss
        dist = self.pc_b - new_pcs
        if self.loss_type == 'L2':
            dist = dist * dist
            loss = np.sum(dist) / point_num
        elif self.loss_type == 'Huber':
            loss = huber_loss(dist, self.huber_limit)
        elif self.loss_type == 'L1':
            dist = np.abs(dist)
            loss = np.sum(dist) / point_num
        return loss
    
    def jac(self, params):
        if self.pc_a is None or self.pc_b is None:
            return np.zeros(params.shape[0])

        # compute motion in intervals
        motion = np.reshape(params, (-1, 4))
        motion = np.sum(motion, axis=0)

        # update to new locations
        x, y, z, theta = motion
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta),  np.cos(theta), 0],
                                    [0,              0,             1]])
        new_pcs = rotation_matrix @ (self.pc_a).T
        new_pcs = new_pcs.T[:, :3]
        new_pcs += np.array([x, y, z])
        dist = self.pc_b - new_pcs

        if self.loss_type == 'L2':
            # Derivative of x, y
            derive = -2 * np.average(dist[:, :3], axis=0)
            derive_x, derive_y, derive_z = derive
    
            # Derivative of theta
            tmp_transformation = np.array([[ np.sin(theta), -np.cos(theta)], 
                                           [ np.cos(theta),  np.sin(theta)]])
            derive_theta_pcs = self.pc_a[:, :2] @ tmp_transformation
            # here we get [N * [xsin + ycos, -xcos + ysin]]
            derive_theta = np.sum(dist[:, :2] * derive_theta_pcs, axis=1)
            derive_theta = np.average(derive_theta)
            derive_theta *= 2
            derivative = np.array([derive_x, derive_y, derive_z, derive_theta])
        elif self.loss_type == 'Huber':
            derivative = huber_jac(theta, dist, self.pc_a, self.huber_limit)
        elif self.loss_type == 'L1':
            sign = np.sign(dist[:, :3])
            derivative = -1 * np.average(sign, axis=0)
            derive_x, derive_y, derive_z = derivative

            mask_xy = sign[:, :2]
            tmp_transformation = np.array([[ np.sin(theta), -np.cos(theta)], 
                                           [ np.cos(theta),  np.sin(theta)]])
            derive_theta_pcs = self.pc_a[:, :2] @ tmp_transformation
            derivative_theta = np.sum(mask_xy * derive_theta_pcs, axis=1)
            derive_theta = np.average(derivative_theta)
            derivative = np.array([derive_x, derive_y, derive_z, derive_theta])

        derivative = np.tile(derivative, self.frame_range[1] - self.frame_range[0])
        return derivative


@numba.njit
def huber_loss(dist, huber_limit):
    result = np.zeros((dist.shape[0], 3), dtype=np.float64)
    for i in range(dist.shape[0]):
        for j in range(3):
            val = dist[i, j]
            if np.abs(val) <= huber_limit:
                result[i, j] = val * val
            else:
                result[i, j] = 2 * huber_limit * (np.abs(val) - huber_limit / 2)
    loss = np.sum(result) / dist.shape[0]
    return loss * (0.5 / huber_limit)


@numba.njit
def huber_jac(theta, dist, pca, huber_limit):
    result = np.zeros((dist.shape[0], 4), dtype=np.float64)
    
    sin_val = np.sin(theta)
    cos_val = np.cos(theta)

    for i in range(dist.shape[0]):
        # x
        val = dist[i, 0]
        theta_der = sin_val * pca[i, 0] + cos_val * pca[i, 1]
        if np.abs(val) <= huber_limit:
            result[i, 0] = -2 * val
            result[i, 3] += 2 * theta_der * val
        else:
            result[i, 0] = -2 * huber_limit * np.sign(val)
            result[i, 3] += -result[i, 0] * theta_der
        # y
        val = dist[i, 1]
        theta_der = -cos_val * pca[i, 0] + sin_val * pca[i, 1]
        if np.abs(val) <= huber_limit:
            result[i, 1] = -2 * val
            result[i, 3] += 2 * theta_der * val
        else:
            result[i, 1] = -2 * huber_limit * np.sign(val)
            result[i, 3] += -result[i, 1] * theta_der
        # z
        val = dist[i, 2]
        if np.abs(val) <= huber_limit:
            result[i, 2] = -2 * val
        else:
            result[i, 2] = -2 * huber_limit * np.sign(val)
    
    derivative = np.sum(result, axis=0) / result.shape[0]
    return derivative * (0.5 / huber_limit)
