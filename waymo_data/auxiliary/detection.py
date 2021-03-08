""" Extract the detections from .bin files
"""
import os, math, numpy as np, itertools, argparse, json
import tensorflow.compat.v1 as tf
from google.protobuf.descriptor import FieldDescriptor as FD
tf.enable_eager_execution()
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='public')
parser.add_argument('--det_folder', type=str, default='../../../datasets/waymo/sot/detection/')
parser.add_argument('--file_name', type=str, default='val.bin')
parser.add_argument('--data_folder', type=str, default='../../../datasets/waymo/sot/')
args = parser.parse_args()

args.output_folder = os.path.join(args.det_folder, args.name, 'dets')
if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)


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
        box_dict['height'],
        box_dict['score']
    ])
    return result


def str_list_to_int(lst):
    result = []
    for t in lst:
        try:
            t = int(t)
            result.append(t)
        except:
            continue
    return result


def main(name, data_folder, det_folder, file_name, out_folder):
    # load timestamp and segment names
    ts_info_folder = os.path.join(data_folder, 'ts_info')
    ts_files = os.listdir(ts_info_folder)
    ts_info = dict()
    segment_name_list = list()
    for ts_file_name in ts_files:
        ts = json.load(open(os.path.join(ts_info_folder, ts_file_name), 'r'))
        segment_name = ts_file_name.split('.')[0]
        ts_info[segment_name] = ts
        segment_name_list.append(segment_name)
    
    # load detection file
    det_folder = os.path.join(det_folder, name)
    f = open(os.path.join(det_folder, file_name), 'rb')
    objects = metrics_pb2.Objects()
    objects.ParseFromString(f.read())
    f.close()
    
    # parse and aggregate detections
    objects = objects.objects
    object_num = len(objects)

    result_bbox, result_type = dict(), dict()
    for seg_name in ts_info.keys():
        result_bbox[seg_name] = dict()
        result_type[seg_name] = dict()
    
    for _i in range(object_num):
        instance = objects[_i]
        segment = instance.context_name
        time_stamp = instance.frame_timestamp_micros

        box = instance.object.box
        bbox_dict = {
            'center_x': box.center_x,
            'center_y': box.center_y,
            'center_z': box.center_z,
            'width': box.width,
            'length': box.length,
            'height': box.height,
            'heading': box.heading,
            'score': instance.score
        }
        bbox_array = bbox_dict2array(bbox_dict)
        obj_type = instance.object.type

        val_index = None
        for _j in range(len(segment_name_list)):
            if segment in segment_name_list[_j]:
                val_index = _j
                break
        segment_name = segment_name_list[val_index]

        frame_number = None
        for _j in range(len(ts_info[segment_name])):
            if ts_info[segment_name][_j] == time_stamp:
                frame_number = _j
                break
        
        if str(frame_number) not in result_bbox[segment_name].keys():
            result_bbox[segment_name][str(frame_number)] = list()
            result_type[segment_name][str(frame_number)] = list()
        result_bbox[segment_name][str(frame_number)].append(bbox_array)
        result_type[segment_name][str(frame_number)].append(obj_type)

        if (_i + 1) % 10000 == 0:
            print(_i + 1, ' / ', object_num)
    
    # store in files
    for _i, segment_name in enumerate(segment_name_list):
        dets = result_bbox[segment_name]
        types = result_type[segment_name]
        print('{} / {}'.format(_i + 1, len(segment_name_list)))

        frame_keys = sorted(str_list_to_int(dets.keys()))
        max_frame = max(frame_keys)
        bboxes = list()
        obj_types = list()
        for key in range(max_frame + 1):
            if str(key) in dets.keys():
                bboxes.append(dets[str(key)])
                obj_types.append(types[str(key)])
            else:
                bboxes.append([])
                obj_types.append([])

        np.savez_compressed(os.path.join(out_folder, "{:}.npz".format(segment_name)), 
            bboxes=bboxes, types=obj_types)


if __name__ == '__main__':
    main(args.name, args.data_folder, args.det_folder, args.file_name, args.output_folder)