""" Example of data loader:
    The data loader has to be an iterator:
    Return a dict of frame data
    Users may create the logic of your own data loader
"""
import os, numpy as np


class ExampleLoader:
    def __init__(self, configs, id, segment_name, data_folder, frame_range):
        """ initialize with the path to data 
        Args:
            data_folder (str): root path to your data
        """
        self.configs = configs
        self.id = id
        self.segment = segment_name
        self.data_loader = data_folder
        self.start_frame = frame_range[0]
        self.end_frame = frame_range[1]
        self.cur_frame = self.start_frame

        self.ego_info = np.load(os.path.join(data_folder, 'ego_info', '{:}.npz'.format(segment_name)), 
            allow_pickle=True)
        self.pcs = np.load(os.path.join(data_folder, 'pc', 'clean_pc', '{:}.npz'.format(segment_name)), 
            allow_pickle=True)
        
        self.use_detection = self.configs['data_loader']['detection']
        self.use_terrain = self.configs['data_loader']['terrain']
        self.obj_type = self.configs['data_loader']['obj_type']
        if self.use_terrain:
            self.terrain = np.load(os.path.join(data_folder, 'pc', 'terrain_pc', '{:}.npz'.format(segment_name)),
                allow_pickle=True)['terrain']
        if self.use_detection:
            self.dets = np.load(os.path.join(data_folder, 'detection', 'three', 'dets', '{:}.npz'.format(segment_name)),
                allow_pickle=True)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.cur_frame > self.end_frame:
            raise StopIteration

        result = dict()
        result['pc'] = self.pcs[str(self.cur_frame)]
        result['ego'] = self.ego_info[str(self.cur_frame)]
        result['terrain'] = None
        result['dets'] = None

        if self.use_terrain:
            result['terrain'] = self.terrain
        if self.use_detection:
            bboxes = self.dets['bboxes'][self.cur_frame]
            inst_types = self.dets['types'][self.cur_frame]
            result['dets'] = [bboxes[i] for i, inst_type in enumerate(inst_types) if inst_type == self.obj_type]

        self.cur_frame += 1
        return result
