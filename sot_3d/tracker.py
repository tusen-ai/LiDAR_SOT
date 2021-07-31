""" Single Object Tracker
"""
import os, numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt 
from . import FrameData, OptimData
from .loss_func import LossFunc
from .motion_model import MotionModel
from .shape_map import ShapeMap
from .data_protos import BBox
from .data_buffer import DataBuffer
from .finder import Finder


class Tracker:
    def __init__(self, id, configs, start_bbox, start_frame, track_len):
        # tracklet configuration
        self.id = id
        self.configs = configs
        self.start_bbox = start_bbox
        self.start_frame = start_frame
        self.track_len = track_len

        # optimization configutation
        self.optim_method = configs['optim']['method']
        self.iter_num = configs['optim']['iter']
        self.optim_options = configs['optim']['options']

        # tracker information
        self.sliding_window_size = configs['running']['window_size']

        # components
        self.data_buffer = DataBuffer(configs)
        self.loss_func = LossFunc(configs)
        self.motion_model = MotionModel(configs)
        self.shape_map = ShapeMap(configs)

        # results
        self.result_log = dict()

        # temporary information
        self.frame_number = start_frame  # frame number in sequence
        self.cur_frame = 0               # frame number in tracklet

        self.input_data = None           # for debug
    
    def track(self, input_data: FrameData):
        """ Perform a frame of tracking
        Args:
            input_data (FrameData): input data

        Returns:
            frame_result: a dict, refer to get_frame_result.
        """
        # prepare for the frame level optimization
        # main job is to inform and register the incoming frame data for each component
        self.pre_frame_optim(input_data)

        # iterative optimization
        if self.cur_frame > 0:
            self.optim_block()
        
        frame_result = self.post_frame_optim()
        if frame_result:
            self.result_log[self.cur_frame] = frame_result
        self.cur_frame += 1
        return frame_result
    
    def optim_block(self):
        """ The actual optimization of loss factors
            Iteratively build up the loss functions (self.pre_optim_step)
            Solve the loss function (minimize)
            Then clean up the mess, such as update the newest motion information (self.post_optim_step)
        """
        ITER_NUM = 0
        while ITER_NUM < self.iter_num:
            # prepare for the single iteration step of optimization
            self.pre_optim_step()
            step_motion = np.zeros(4 * (self.motion_model.size - 1))

            # steps of optimization
            res = minimize(self.loss_func.loss, step_motion, method=self.optim_method,
                jac=self.loss_func.jac, options=self.optim_options)
            
            # clean up the effect from optimization
            self.optim_motion = res.x
            self.post_optim_step()
            ITER_NUM += 1
        return
    
    def pre_frame_optim(self, input_data: FrameData):
        """ The frame-level preparation for loss functions.
            These operations are out-of-the-loop of iterative optimization.
            Including:
                * motion model predict the motion
                * shape map initialization
                etc...
        Args:
            input_data (FrameData): input data
        """
        if self.cur_frame == 0:
            self.tmp_init_box0 = input_data.start_bbox
            self.tmp_init_pc0 = input_data.pc        
            pred_motion = np.zeros(4)

            # initialize the starting location of motion model here
            self.motion_model.set_start_location(input_data.start_bbox)

        elif self.cur_frame == 1:
            self.tmp_init_pc1 = input_data.pc
            finder = Finder(self.configs['finder'], self.tmp_init_box0, self.tmp_init_pc0, self.tmp_init_pc1)
            pred_motion = finder.icp()
            del self.tmp_init_pc0, self.tmp_init_pc1
        else:
            pred_motion = self.motion_model.predict_motion() 
        self.motion_model.pre_frame_optim(pred_motion)
        self.shape_map.pre_frame_optim(input_data)
        self.data_buffer.pre_frame_optim(input_data)
        self.loss_func.pre_frame_optim(input_data)
        self.input_data = input_data
        return
    
    def pre_optim_step(self):
        """ Preparation operations before each step of optimization.
            Extract related data from: motion model, data buffer, shape model
            Then build up the optim_data, feed it into loss_func to build up the loss function
        """
        # bbox and motion
        optim_data_motion_model = self.motion_model.pre_optim_step()

        # get the pc and ego information
        optim_data_data_buffer = self.data_buffer.pre_optim_step()
        
        # get the pred states and shape information
        pred_states = np.zeros((self.motion_model.size, 4))
        for i in range(self.motion_model.size):
            pred_bbox = optim_data_motion_model[i]['bbox']
            pred_states[i, :] = BBox.bbox2array(pred_bbox)[:4]
        optim_data_shape_map = self.shape_map.pre_optim_step(pred_states)

        # merge them and create a frame data
        optim_data_list = [{**optim_data_data_buffer[i],
                           **optim_data_motion_model[i],
                           **optim_data_shape_map[i]} 
                           for i in range(len(optim_data_motion_model))]
        optim_data = OptimData.dict2optim_data(optim_data_list)

        # send the optim data into the loss factors
        self.loss_func.pre_optim_step(optim_data)
        return
    
    def post_optim_step(self):
        self.motion_model.post_optim_step(self.optim_motion)
        self.loss_func.post_optim_step()
        self.shape_map.post_optim_step()
        self.data_buffer.post_optim_step()
        return
    
    def post_frame_optim(self):
        self.motion_model.post_frame_optim()

        pc = self.data_buffer.pc_buffer.last()
        optim_bbox = self.motion_model.state_buffer.last()
        self.shape_map.post_frame_optim(pc, optim_bbox)

        frame_result = None
        if self.cur_frame > 0:
            frame_result = self.get_frame_result()
        return frame_result
    
    def get_frame_result(self):
        """ Frame_Result format:
            When optimizing a sliding window of size N
            {
                0:   Estimated result 0 between frames 0 - 1,
                1:   Estimated result 1 between frames 1 - 2,
                ...
                N-2: Estimated result 1 between frames N-2 - N-1
            }
            Each Estimated result is another dict
            {
                'bbox0': BBox Object,
                'bbox1': BBox Object,
                'motion': [x, y, z, theta]
            }
        """
        frame_result = dict()
        for i in range(1, self.motion_model.size):
            frame_result[i - 1] = dict()
            bboxa = self.motion_model.state_buffer.access(i - 1)
            bboxb = self.motion_model.state_buffer.access(i)
            motion = self.motion_model.motion_buffer.access(i)

            frame_result[i - 1]['bbox0'] = bboxa
            frame_result[i - 1]['bbox1'] = bboxb
            frame_result[i - 1]['motion'] = motion.tolist()

        return frame_result
    
    def summary_result(self):
        """ return the result of the whole tracklet
            {
                1: estimated result on frame 1,
                2: estimated result on frame 2,
                ...
                n: estimated result on frame n,
            }
        """
        tracklet_result = dict()
        for i in range(1, self.cur_frame):
            frame_result = self.result_log[i]
            
            # take the latest result for online setting by default
            latest_key = max(list(frame_result.keys()))
            latest_result = frame_result[latest_key]
            bbox0 = BBox.bbox2array(latest_result['bbox0']).tolist()
            bbox1 = BBox.bbox2array(latest_result['bbox1']).tolist()
            tracklet_result[i] = {
                'motion': latest_result['motion'],
                'bbox0': bbox0,
                'bbox1': bbox1}
        return tracklet_result   
