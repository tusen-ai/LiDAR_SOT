""" Detection Factor.
    Involve the detection factors in the optimization process.
"""
import numpy as np, numba
from ..data_protos import BBox
from .. import utils
from .. import FrameData, OptimData


class DetectionFactor:
    def __init__(self, configs):
        self.score_threshold = configs['score_threshold']
        self.iou_threshold = configs['iou_threshold']
        self.prev_bbox = None     # previous frame bbox
        self.bbox_size = None     # size of bbox
        self.tgt_det = None       # target detection bbox

    def pre_frame_optim(self, input_data: FrameData):
        # on the first frame, get the target size
        start_bbox = input_data.start_bbox
        if start_bbox is not None:
            self.bbox_size = np.array([start_bbox.l, start_bbox.w, start_bbox.h])
        return
    
    def pre_optim_step(self, optim_data: OptimData, frame_indexes):
        """ Preprare the loss computation for detection factor.
            Select the detection bbox.
        """
        # use previous frame bbox to compute the motion
        self.prev_bbox = optim_data.bboxes[frame_indexes[0]]
        pred_bbox = optim_data.bboxes[frame_indexes[1]]
        
        # select the current frame dets by score
        cur_frame_dets = optim_data.dets[frame_indexes[1]]
        ego = optim_data.egos[frame_indexes[1]]
        cur_frame_dets = [det for det in cur_frame_dets 
            if det.s > self.score_threshold]
        if len(cur_frame_dets) == 0:
            self.tgt_det = None
            return

        # rectify its corners
        ego_location = ego[:3, 3]
        cur_frame_dets = [utils.bbox_rectify(det, self.bbox_size, ego_location)
            for det in cur_frame_dets]
        
        # select the bbox with largest iou
        ious = [utils.iou2d(det, pred_bbox) for det in cur_frame_dets]
        optim_det_index = np.argmax(ious)
        if ious[optim_det_index] >= self.iou_threshold:
            self.tgt_det = cur_frame_dets[optim_det_index]
        else:
            self.tgt_det = None
        return
    
    def post_optim_step(self):
        return
    
    def post_frame_optim(self):
        return
    
    def loss(self, params):
        if self.tgt_det is None:
            return 0
        det_bbox_array = BBox.bbox2array(self.tgt_det)[:4]
        prev_bbox_array = BBox.bbox2array(self.prev_bbox)[:4]
        dist = prev_bbox_array + params - det_bbox_array
        loss = np.sum(dist * dist)
        return loss
    
    def jac(self, params):
        if self.tgt_det is None:
            return np.zeros_like(params)
        det_bbox_array = BBox.bbox2array(self.tgt_det)[:4]
        prev_bbox_array = BBox.bbox2array(self.prev_bbox)[:4]
        dist = prev_bbox_array + params - det_bbox_array
        derivative = 2 * dist
        return derivative
