""" The class representing the information of a tracklet
"""


class Tracklet:
    def __init__(self, id, obj_type, start_bbox, start_frame, pc_num):
        self.id = id
        self.segment_name = ''
        self.obj_type = obj_type
        self.start_frame = start_frame
        self.end_frame = start_frame
        self.bboxes = [start_bbox]
        self.pc_nums = [pc_num]
        return
    
    def set_new_frame(self, bbox, pc_num, frame_idx):
        self.end_frame = frame_idx
        self.bboxes.append(bbox)
        self.pc_nums.append(pc_num)
        return
    
    def tracklet2dict(self):
        return {
            'id': self.id,
            'type': self.obj_type,
            'segment_name': self.segment_name,
            'frame_range': (self.start_frame, self.end_frame),
        }