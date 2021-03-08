import numpy as np
from tracklet import Tracklet


def seq2tracklets(bboxes, ids, types, pc_nums, type_for_selection=[1]):
    """ Reformat the gt sequence to tracklets
    Args:
        bboxes ([list of arrays]): the [x, y, z, heading, length, width, height] of bboxes
        ids ([type]): the ids of bboxes
        types ([type]): the types of bboxes
        pc_nums ([type]): lidar point number of bboxes
        type_for_selection: types of object for selection, default to vehicle
    Return:
        list of tracklets and their ids
    """
    frame_num = len(bboxes)
    buffer_tracklets = list()
    buffer_ids = list()
    result_tracklets = list()
    result_ids = list()
    for frame_index in range(frame_num):
        frame_bboxes, frame_ids, frame_types, frame_pc_nums = \
            bboxes[frame_index], ids[frame_index], types[frame_index], pc_nums[frame_index]
    
        bbox_num = len(frame_bboxes)
        for i in range(bbox_num):
            bbox, id, obj_type, pc_num = frame_bboxes[i], frame_ids[i], frame_types[i], frame_pc_nums[i]
            
            # select the vehicle type
            if obj_type not in type_for_selection:
                continue

            # update the buffers with the newest information
            if id not in buffer_ids and id not in result_ids:
                buffer_ids.append(id)
                buffer_tracklets.append(Tracklet(id, obj_type, bbox, frame_index, pc_num))
            elif id in buffer_ids:
                trackelt_index = buffer_ids.index(id)
                buffer_tracklets[trackelt_index].set_new_frame(bbox, pc_num, frame_index)
        
        # put the closed tracklets to result
        removed_index = list()
        for tracklet_index, tracklet_id in enumerate(buffer_ids):
            if buffer_tracklets[tracklet_index].end_frame != frame_index:
                result_ids.append(tracklet_id)
                result_tracklets.append(buffer_tracklets[tracklet_index])
                removed_index.append(tracklet_index)
        
        # clean up the buffer
        removed_index = reversed(removed_index)
        for index in removed_index:
            buffer_ids.pop(index)
            buffer_tracklets.pop(index)
    
    # when the tracklet finish, there might be some left tracklets
    for i, tracklet in enumerate(buffer_tracklets):
        result_ids.append(tracklet.id)
        result_tracklets.append(tracklet)

    return result_ids, result_tracklets
