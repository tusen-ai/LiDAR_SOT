import numpy as np
import sot_3d, sot_3d.factors as factors, sot_3d.utils as utils
import matplotlib.pyplot as plt


class LossFunc:
    def __init__(self, configs):
        self.configs = configs
        self.window_size = self.configs['running']['window_size']
        self.size = 0
        self.factor_options = self.configs['factors']
        self.switches = self.factor_options['switch']

        # initialize factors
        self.icp_factors = [factors.ICPLoss(self.configs['factors']['icp_loss']) 
            for i in range(self.window_size - 1)]
        self.motion_prior = factors.MotionPriorFactor(self.configs['factors']['motion_prior'])
        self.shape_factors = [factors.ShapeLoss(self.configs['factors']['shape_loss'])
            for i in range(self.window_size - 1)]
        self.latitude_factors = [factors.LatitudeLoss(self.configs['factors']['latitude'])
            for i in range(self.window_size - 1)]
        self.motion_consistency = [factors.MotionConsistency(self.configs['factors']['motion_consistency'])
            for i in range(self.window_size - 1)]
        self.detections = [factors.DetectionFactor(self.configs['factors']['detection'])
            for i in range(self.window_size - 1)]

        # weight
        self.icp_weight = configs['weight']['icp']
        self.motion_prior_weight = configs['weight']['motion_prior']
        self.shape_loss_weight = configs['weight']['shape_loss']
        self.latitude_loss_weight = configs['weight']['latitude']
        self.detection_weight = configs['weight']['detection']
        self.motion_consistency_weight = configs['weight']['motion_consistency']

    def pre_frame_optim(self, input_data: sot_3d.FrameData):
        self.size = min(self.size + 1, self.window_size)
        for i in range(self.window_size - 1):
            if self.switches['latitude']:
                self.latitude_factors[i].pre_frame_optim(input_data)
            if self.switches['detection']:
                self.detections[i].pre_frame_optim(input_data)
    
    def pre_optim_step(self, optim_data: sot_3d.OptimData):
        for i in range(self.size - 1):
            if self.switches['icp_loss']:
                self.icp_factors[i].pre_optim_step(optim_data, (i, i + 1))
            if self.switches['motion_consistency']:
                self.motion_consistency[i].pre_optim_step(optim_data, (i, i + 1))
            if self.switches['shape_loss']:
                self.shape_factors[i].pre_optim_step(optim_data, (i, i + 1))
            if self.switches['latitude']:
                self.latitude_factors[i].pre_optim_step(optim_data, (i, i + 1))
            if self.switches['detection']:
                self.detections[i].pre_optim_step(optim_data, (i, i + 1))
        self.motion_prior.pre_optim_step(optim_data, (0, self.size - 1))
        return
    
    def post_optim_step(self):
        for i in range(self.size - 1):
            if self.switches['icp_loss']:
                self.icp_factors[i].post_optim_step()
            if self.switches['motion_consistency']:
                self.motion_consistency[i].post_optim_step()
            if self.switches['shape_loss']:
                self.shape_factors[i].post_optim_step()
            if self.switches['latitude']:
                self.latitude_factors[i].post_optim_step()
            if self.switches['detection']:
                self.detections[i].post_optim_step()
        self.motion_prior.post_optim_step()
    
    def post_frame_optim(self):
        for i in range(self.size - 1):
            if self.switches['icp_loss']:
                self.icp_factors[i].post_frame_optim()
            if self.switches['motion_consistency']:
                self.motion_consistency[i].post_frame_optim()
            if self.switches['shape_loss']:
                self.shape_factors[i].post_frame_optim()
            if self.switches['latitude']:
                self.latitude_factors[i].post_frame_optim()
            if self.switches['detection']:
                self.detections[i].post_frame_optim()
        self.motion_prior.post_frame_optim()
    
    def loss(self, params):
        result = 0
        
        # icp loss
        if self.switches['icp_loss']:
            icp_loss = 0
            for i in range(self.size - 1):
                icp_loss += self.icp_factors[i].loss(params[4 * i: 4 * (i + 1)])
            icp_loss /= (self.size - 1)
            result += icp_loss * self.icp_weight 

        # motion prior
        motion_prior_loss = self.motion_prior.loss(params[:])
        result += motion_prior_loss * self.motion_prior_weight

        # shape loss
        if self.switches['shape_loss']:
            shape_loss = 0
            for i in range(self.size - 1):
                shape_loss += self.shape_factors[i].loss(params[4 * i: 4 * (i + 1)])
            shape_loss /= (self.size - 1)
            result += shape_loss * self.shape_loss_weight

        # latitude
        if self.switches['latitude']:
            latitude_loss = 0
            for i in range(self.size - 1):
                latitude_loss += self.latitude_factors[i].loss(params[4 * i: 4 * (i + 1)])
            latitude_loss /= (self.size - 1)
            result += latitude_loss * self.latitude_loss_weight
        
        # detections
        if self.switches['detection']:
            det_loss = 0
            for i in range(self.size - 1):
                det_loss += self.detections[i].loss(params[4 * i: 4 * (i + 1)])
            det_loss /= (self.size - 1)
            result += det_loss * self.detection_weight

        # motion consistency
        if self.switches['motion_consistency']:
            motion_consistency_loss = 0
            for i in range(self.size - 1):
                motion_consistency_loss += self.motion_consistency[i].loss(params[4 * i: 4 * (i + 1)])
            motion_consistency_loss /= (self.size - 1)
            result += motion_consistency_loss * self.motion_consistency_weight
        
        return result
    
    def jac(self, params):
        result = 0
        
        # icp
        if self.switches['icp_loss']:
            icp_der = np.zeros(params.shape[0])
            for i in range(self.size - 1):
                icp_der[4 * i: 4 * (i + 1)] += \
                    self.icp_factors[i].jac(params[4 * i: 4 * (i + 1)])
            icp_der /= (self.size - 1)
            result += icp_der

        # motion prior
        motion_prior_der = self.motion_prior.jac(params[:])
        result += motion_prior_der * self.motion_prior_weight

        # shape loss
        if self.switches['shape_loss']:
            shape_der = np.zeros(params.shape[0])
            for i in range(self.size - 1):
                shape_der[4 * i: 4 * (i + 1)] += \
                    self.shape_factors[i].jac(params[4 * i: 4 * (i + 1)])
            shape_der /= (self.size - 1)
            result += shape_der * self.shape_loss_weight

        # latitude loss
        if self.switches['latitude']:
            latitude_der = np.zeros(params.shape[0])
            for i in range(self.size - 1):
                latitude_der[4 * i: 4 * (i + 1)] += \
                    self.latitude_factors[i].jac(params[4 * i: 4 * (i + 1)])
            latitude_der /= (self.size - 1)
            result += latitude_der * self.latitude_loss_weight
        
         # detections
        if self.switches['detection']:
            det_der = np.zeros(params.shape[0])
            for i in range(self.size - 1):
                det_der[4 * i: 4 * (i + 1)] += \
                    self.detections[i].jac(params[4 * i: 4 * (i + 1)])
            det_der /= (self.size - 1)
            result += det_der * self.detection_weight
        
        # motion consistency
        if self.switches['motion_consistency']:
            motion_consistency_der = np.zeros(params.shape[0])
            for i in range(self.size - 1):
                motion_consistency_der[4 * i: 4 * (i + 1)] += \
                    self.motion_consistency[i].jac(params[4 * i: 4 * (i + 1)])
            motion_consistency_der /= (self.size - 1)
            result += motion_consistency_der * self.motion_consistency_weight
        
        return result
