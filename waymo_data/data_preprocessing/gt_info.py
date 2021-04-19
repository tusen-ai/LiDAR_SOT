""" Extract the gt information
    Output format compressed .npz files
    {
        'bboxes': bbox sequence,
        'ids'   : IDS
        'pc_num': Number of LiDAR points
        'types' : Type of objects
    }
    Inside each sequence, its format is a list.
    Take BBox for example, each box is an array [x, y, z, heading, length, width, height]
    [
        [BBox00, BBox01, BBox02...],  --> BBoxes in frame 0,
        [BBox10, BBox11, BBox12...],  --> BBoxes in frame 1
        ...
    ]
    Note that the order in bboxes, ids, types, pc_nums are identical.
"""
import argparse
import math
import numpy as np
import json
import os
from google.protobuf.descriptor import FieldDescriptor as FD
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()
import multiprocessing
import utils

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', type=str, default='../../../datasets/waymo/validation/')
parser.add_argument('--output_folder', type=str, default='../../../datasets/waymo/sot/')
parser.add_argument('--process', type=int, default=1)
args = parser.parse_args()

args.clean_pc_folder = os.path.join(args.output_folder, 'pc', 'clean_pc')
args.output_folder = os.path.join(args.output_folder, 'gt_info')
if not os.path.join(args.output_folder):
    os.makedirs(args.output_folder)


def pb2dict(obj):
    """
    Takes a ProtoBuf Message obj and convertes it to a dict.
    """
    adict = {}
    # if not obj.IsInitialized():
    #     return None
    for field in obj.DESCRIPTOR.fields:
        if not getattr(obj, field.name):
            continue
        if not field.label == FD.LABEL_REPEATED:
            if not field.type == FD.TYPE_MESSAGE:
                adict[field.name] = getattr(obj, field.name)
            else:
                value = pb2dict(getattr(obj, field.name))
                if value:
                    adict[field.name] = value
        else:
            if field.type == FD.TYPE_MESSAGE:
                adict[field.name] = [pb2dict(v) for v in getattr(obj, field.name)]
            else:
                adict[field.name] = [v for v in getattr(obj, field.name)]
    return adict


def bbox_dict2array(box_dict):
    """transform box dict in waymo_open_format to array
    Args:
        box_dict ([dict]): waymo_open_dataset formatted bbox
    """
    result = np.array([
        box_dict['center_x'],
        box_dict['center_y'],
        box_dict['center_z'],
        box_dict['heading'],
        box_dict['length'],
        box_dict['width'],
        box_dict['height']
    ])
    return result


def main(data_folder, output_folder, pc_folder, token=0, process_num=1):
    tf_records = os.listdir(data_folder)
    tf_records = [x for x in tf_records if 'tfrecord' in x]
    tf_records = sorted(tf_records) 
    for record_index, tf_record_name in enumerate(tf_records[:]):
        if record_index % process_num != token:
            continue
        print('starting for gt info ', record_index + 1, ' / ', len(tf_records), ' ', tf_record_name)
        FILE_NAME = os.path.join(data_folder, tf_record_name)
        dataset = tf.data.TFRecordDataset(FILE_NAME, compression_type='')
        segment_name = tf_record_name.split('.')[0]
        clean_pc = np.load(os.path.join(pc_folder, '{:}.npz'.format(segment_name)), allow_pickle=True)

        frame_num = 0
        sequence_bboxes = list()
        IDS = list()
        inst_types = list()
        lidar_pc_nums = list()

        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            frame_pc = clean_pc[str(frame_num)]
     
            # bbox arrays
            laser_labels = frame.laser_labels
            frame_ids = list()
            frame_boxes = list()
            frame_types = list()
            frame_nums = list()
            for laser_label in laser_labels:
                id = laser_label.id
                box = laser_label.box
                box_dict = pb2dict(box)
                box_array = bbox_dict2array(box_dict)
                frame_boxes.append(box_array[np.newaxis, :])
                frame_ids.append(id)
                frame_types.append(laser_label.type)

                # the number of lidar points of object, exclude ground points using the clean pcs
                lidar_pcs_in_box = laser_label.num_lidar_points_in_box
                lidar_pcs_in_box = utils.pc_in_box(box_array, frame_pc, 1.0).shape[0]
                frame_nums.append(lidar_pcs_in_box)
            
            if len(frame_boxes) > 0:
                frame_boxes = np.concatenate(frame_boxes, axis=0)
            sequence_bboxes.append(frame_boxes)
            IDS.append(frame_ids)
            inst_types.append(frame_types)
            lidar_pc_nums.append(frame_nums)

            frame_num += 1
            if frame_num % 10 == 0:
                print('file {:} / {:} frame number {:}'.format(record_index + 1, len(tf_records), frame_num))
        print('{:} frames in total'.format(frame_num))
        
        sequence_bboxes = np.array(sequence_bboxes)
        IDS = np.array(IDS)
        inst_types = np.array(inst_types)

        np.savez_compressed(os.path.join(output_folder, "{}.npz".format(segment_name)), 
            bboxes=sequence_bboxes, ids=IDS, types=inst_types, pc_nums=lidar_pc_nums)


if __name__ == '__main__':
    if args.process > 1:
        pool = multiprocessing.Pool(args.process)
        for token in range(args.process):
            result = pool.apply_async(main, args=(args.data_folder, args.output_folder, args.clean_pc_folder, token, args.process))
        pool.close()
        pool.join()
    else:
        main(args.data_folder, args.output_folder, args.clean_pc_folder, 0, 1)
