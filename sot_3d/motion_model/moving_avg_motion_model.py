""" An example moving average motion model
"""
import numpy as np, math, copy
from .. import utils
from ..data_protos import BBox


class MotionModel:
    def __init__(self, configs):
        self.configs = configs
        self.window_size = self.configs['running']['window_size']
        self.moving_avg_weight = self.configs['motion_model']['moving_avg_weight']
        self.size = 0
        self.cur_frame = 0
        
        self.start_state = None
        self.moving_avg_value = None
        # information in the sliding window
        self.motion_buffer = utils.CircularBuffer(self.window_size, 0, 0)
        self.state_buffer = utils.CircularBuffer(self.window_size, 0, 0)
    
    def set_start_location(self, start_bbox: BBox):
        self.start_state = start_bbox
        return
    
    def predict_motion(self):
        """ predict a rough prediction of the motion, currently by moving_avg
        """
        return self.moving_avg_value
    
    def pre_frame_optim(self, pred_motion):
        """ before optimizing each frame, motion model takes in a roughly predicted motion
        Args:
            pred_motion (np.ndarray): a rough prediction for the motion
        """
        # When encountering frame 0
        # push the start_bbox and the dummy pred_motion into the buffer
        if self.size == 0:
            self.state_buffer.push(self.start_state)
            self.motion_buffer.push(pred_motion)
        # At other frames, the predicted motion is right
        else:
            prev_bbox = self.state_buffer.last()
            pred_bbox = BBox.motion2bbox(prev_bbox, pred_motion)
            self.state_buffer.push(pred_bbox)
            self.motion_buffer.push(pred_motion)
        
        self.size = self.state_buffer.size
        return
    
    def pre_optim_step(self):
        """ Before each optimization step, the motion model prepare the following motion related information
            1. rough predicted motion from the motion model: as a reference
            2. the location of bboxes in the current state of optimization
            Motion model return a list of dict. [{'motion': xxx, 'bbox': xxx}, ..., ...]
            Each item corresponds to the data of a frame
        """
        result = [dict() for i in range(self.size)]
        for i in range(self.size):
            result[i]['motion'] = self.motion_buffer.access(i)
            result[i]['bbox'] = self.state_buffer.access(i)
        return result
    
    def post_optim_step(self, optim_motions):
        """ When each time an optimized motion arrive, we have to update the corresponding motion and predicted bbox location
        """
        if len(optim_motions.shape) == 1:
            optim_motions = optim_motions.reshape((-1, 4))
        for i in range(1, self.size):
            self.motion_buffer.set_val(i, optim_motions[i - 1])
            prev_bbox = self.state_buffer.access(i - 1)
            next_bbox = BBox.motion2bbox(prev_bbox, optim_motions[i - 1])
            self.state_buffer.set_val(i, next_bbox)
        return
    
    def post_frame_optim(self):
        """ After optimizing for a frame, update the motion model
            In the moving avg case, update the moving avg value
        """
        # add one tick to the frame counter
        self.cur_frame += 1
        # first frame, no need for update the moving avg value
        if self.cur_frame == 1:
            return
        # update the moving average value
        if self.moving_avg_value is None:
            self.moving_avg_value = self.motion_buffer.last()
        else:
            newest_motion = self.motion_buffer.last()
            self.moving_avg_value = self.moving_avg_weight * self.moving_avg_value + \
                (1 - self.moving_avg_weight) * newest_motion
        return