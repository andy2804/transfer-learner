"""
author: az
"""
import os

import numpy as np
import tensorflow as tf

from rosbag_converter import (ImageInstance,
                              ObjectDetected)
from objdetection.meta.datasets.encoder import Encoder


class TFRecordsEncoder(Encoder):
    def __init__(self):
        super().__init__()

    def _consolidate_features(self, parsed_features):
        consolidated_features = {}
        for key, value in parsed_features.items():
            topic = os.path.dirname(key)
            element = os.path.basename(key)
            if not topic in consolidated_features:
                consolidated_features[topic] = {}
            consolidated_features[topic][element] = value

        return consolidated_features

    def encode(self, instance):
        """
        Build example proto from DataInstance
        :param instance:
        :return: tf.train.Example
        """

        # todo save features to yaml / json file so tfRecord can easily be decoded
        features = {}
        for key, value in instance.items():
            print(key)
            if isinstance(value, ImageInstance):
                image_format = b'png'
                features[key + '/image'] = self._bytes_feature(value.image.tostring())
                features[key + '/format'] = self._bytes_feature(image_format)
                features[key + '/shape'] = self._int64_feature(list(value.image.shape))
                features[key + '/ts'] = self._int64_feature(value.ts.to_nsec())

            elif value.__class__.__name__ == 'ndarray':
                if 'events' in key:
                    features[key + '/ts'] = self._int64_feature(value[:, 0].tolist())
                    features[key + '/xyp'] = self._bytes_feature(
                            (value[:, 1:].astype(np.uint8)).tostring())
                    features[key + '/xyp_shape'] = self._int64_feature(list(value[:, 1:].shape))
                elif 'imu' in key:
                    pass
                    # features[key + '/encoded'] = self._bytes_feature(value.tostring())
                    # features[key + '/shape'] = self._float_feature(list(value.shape))
                else:
                    pass

            elif isinstance(value, ObjectDetected):
                x_center, y_center, width, height = self._minmax_to_center_boxes(value.boxes)
                features[key + '/x_center'] = self._float_feature(x_center.tolist())
                features[key + '/y_center'] = self._float_feature(y_center.tolist())
                features[key + '/width'] = self._float_feature(width.tolist())
                features[key + '/height'] = self._float_feature(height.tolist())
                features[key + '/classes'] = self._int64_feature(value.classes.tolist())
                features[key + '/scores'] = self._float_feature(value.scores.tolist())
                features[key + '/ts'] = self._int64_feature(value.ts.to_nsec())

            else:
                # camera info not encoded
                # not relevant for the time being right now
                # elif value.__class__.__name__ == 'CameraInfo':
                #     features[key + '/height'] = self._int64_feature(value.height)
                #     features[key + '/width'] = self._int64_feature(value.width)
                #     features[key + '/distortion'] = self._bytes_feature(
                # value.distortion_model.encode())
                #     features[key + '/D'] = self._float_feature(list(np.ravel(value.D)))
                #     features[key + '/K'] = self._float_feature(list(np.ravel(value.K)))
                #     features[key + '/R'] = self._float_feature(list(np.ravel(value.R)))
                #     features[key + '/P'] = self._float_feature(list(np.ravel(value.P)))
                #     features[key + '/binning_x'] = self._float_feature(list(np.ravel(
                # value.binning_x)))
                #     features[key + '/binning_y'] = self._float_feature(list(np.ravel(
                # value.binning_y)))
                #     features[key + '/roi'] = self._float_feature(list(np.ravel(value.roi)))
                pass

        example = tf.train.Example(features=tf.train.Features(feature=features))
        return example

    def decode(self, instance):
        """
        :param instance:
        :return:
        """
        keys_to_features = {
            '/pylon_rgb/image_raw/image':     tf.FixedLenFeature([], tf.string),
            '/pylon_rgb/image_raw/format':    tf.FixedLenFeature([], tf.string,
                                                                 default_value='png'),
            '/pylon_rgb/image_raw/shape':     tf.FixedLenSequenceFeature(shape=[1],
                                                                         dtype=tf.int64,
                                                                         allow_missing=True),
            '/pylon_rgb/image_raw/timestamp': tf.FixedLenSequenceFeature(shape=[1],
                                                                         dtype=tf.int64,
                                                                         allow_missing=True),
            '/dvs/image_raw/image':           tf.FixedLenFeature([], tf.string),
            '/dvs/image_raw/format':          tf.FixedLenFeature([], tf.string,
                                                                 default_value='png'),
            '/dvs/image_raw/shape':           tf.FixedLenSequenceFeature(shape=[1],
                                                                         dtype=tf.int64,
                                                                         allow_missing=True),
            '/dvs/image_raw/timestamp':       tf.FixedLenSequenceFeature(shape=[1],
                                                                         dtype=tf.int64,
                                                                         allow_missing=True),
            # objects
            '/pylon_rgb/objects/x_center':    tf.FixedLenSequenceFeature(shape=[1],
                                                                         dtype=tf.float32,
                                                                         allow_missing=True),
            '/pylon_rgb/objects/y_center':    tf.FixedLenSequenceFeature(shape=[1],
                                                                         dtype=tf.float32,
                                                                         allow_missing=True),
            '/pylon_rgb/objects/width':       tf.FixedLenSequenceFeature(shape=[1],
                                                                         dtype=tf.float32,
                                                                         allow_missing=True),
            '/pylon_rgb/objects/height':      tf.FixedLenSequenceFeature(shape=[1],
                                                                         dtype=tf.float32,
                                                                         allow_missing=True),
            '/pylon_rgb/objects/classes':     tf.FixedLenSequenceFeature(shape=[1],
                                                                         dtype=tf.int64,
                                                                         allow_missing=True),
            '/pylon_rgb/objects/scores':      tf.FixedLenSequenceFeature(shape=[1],
                                                                         dtype=tf.float32,
                                                                         allow_missing=True),
            '/pylon_rgb/objects/ts':          tf.FixedLenSequenceFeature(shape=[1],
                                                                         dtype=tf.int64,
                                                                         allow_missing=True),
            # events
            '/dvs/events/ts':                 tf.FixedLenSequenceFeature(shape=[1],
                                                                         dtype=tf.int64,
                                                                         allow_missing=True),
            '/dvs/events/xyp':                tf.FixedLenFeature([], tf.string),
            '/dvs/events/xyp_shape':          tf.FixedLenSequenceFeature(shape=[1],
                                                                         dtype=tf.int64,
                                                                         allow_missing=True),

            # imu
            # '/dvs/imu/encoded':               tf.FixedLenFeature([], tf.string),
            # '/dvs/imu/shape':                 tf.FixedLenSequenceFeature(shape=[1],
            #                                                              dtype=tf.int64,
            #                                                             allow_missing=True),
        }

        parsed_features = tf.parse_single_example(instance, features=keys_to_features)
        # parse image
        image = tf.decode_raw(parsed_features['/pylon_rgb/image_raw/image'], tf.uint8)
        shape_im = tf.squeeze(
                tf.cast(parsed_features['/pylon_rgb/image_raw/shape'], tf.int32), [1])
        image = tf.reshape(image, shape_im)
        # parse events
        events = tf.decode_raw(parsed_features['/dvs/events/xyp'], tf.uint8)
        shape_ev = tf.squeeze(
                tf.cast(parsed_features['/dvs/events/xyp_shape'], tf.int32), [1])
        events = tf.cast(tf.reshape(events, shape_ev), dtype=tf.float64)
        events_ts = tf.cast(tf.squeeze(parsed_features['/dvs/events/ts'], [1]), dtype=tf.float32)
        # fixme double check casting
        events = tf.stack([events_ts, events])
        # parse boxes
        x_cent = tf.squeeze(parsed_features['/pylon_rgb/objects/x_center'], [1])
        y_cent = tf.squeeze(parsed_features['/pylon_rgb/objects/y_center'], [1])
        w = tf.squeeze(parsed_features['/pylon_rgb/objects/width'], [1])
        h = tf.squeeze(parsed_features['/pylon_rgb/objects/height'], [1])
        classes = tf.squeeze(
                tf.cast(parsed_features['/pylon_rgb/objects/classes'], tf.int64), [1])
        scores = tf.squeeze(parsed_features['/pylon_rgb/objects/scores'], [1])
        boxes = tf.stack([x_cent, y_cent, w, h], axis=0)
        return {'image':   image,
                'events':  events,
                'boxes':   boxes,
                'classes': classes,
                'scores':  scores
                }


"""
    def decode_andy(self, example):

        # todo anyway to make tfrecord handling nicer?
        keys_to_features = {
            # pylon image
            '/pylon_rgb/image_raw/image':     tf.FixedLenFeature([], tf.string),
            '/pylon_rgb/image_raw/format':    tf.FixedLenFeature([], tf.string,
                                                                 default_value='png'),
            '/pylon_rgb/image_raw/shape':     tf.FixedLenSequenceFeature(shape=[1],
                                                                         dtype=tf.int64,
                                                                         allow_missing=True),
            '/pylon_rgb/image_raw/timestamp': tf.FixedLenSequenceFeature(shape=[1],
                                                                         dtype=tf.int64,
                                                                         allow_missing=True),
            # objects
            '/pylon_rgb/objects/x_center':    tf.FixedLenSequenceFeature(shape=[1],
                                                                         dtype=tf.float32,
                                                                         allow_missing=True),
            '/pylon_rgb/objects/y_center':    tf.FixedLenSequenceFeature(shape=[1],
                                                                         dtype=tf.float32,
                                                                         allow_missing=True),
            '/pylon_rgb/objects/width':       tf.FixedLenSequenceFeature(shape=[1],
                                                                         dtype=tf.float32,
                                                                         allow_missing=True),
            '/pylon_rgb/objects/height':      tf.FixedLenSequenceFeature(shape=[1],
                                                                         dtype=tf.float32,
                                                                         allow_missing=True),
            '/pylon_rgb/objects/scores':      tf.FixedLenSequenceFeature(shape=[1],
                                                                         dtype=tf.float32,
                                                                         allow_missing=True),
            '/pylon_rgb/objects/label':       tf.FixedLenSequenceFeature(shape=[1],
                                                                         dtype=tf.int64,
                                                                         allow_missing=True),
            '/dvs/image_raw/image':           tf.FixedLenFeature([], tf.string),
            '/dvs/image_raw/format':          tf.FixedLenFeature([], tf.string,
                                                                 default_value='png'),
            '/dvs/image_raw/shape':           tf.FixedLenSequenceFeature(shape=[1],
                                                                         dtype=tf.int64,
                                                                         allow_missing=True),
            '/dvs/image_raw/timestamp':       tf.FixedLenSequenceFeature(shape=[1],
                                                                         dtype=tf.int64,
                                                                         allow_missing=True),
            '/dvs/events/encoded':            tf.FixedLenFeature([], tf.string),
            '/dvs/events/shape':              tf.FixedLenSequenceFeature(shape=[1],
                                                                         dtype=tf.int64,
                                                                         allow_missing=True),
            '/dvs/imu/encoded':               tf.FixedLenFeature([], tf.string),
            '/dvs/imu/shape':                 tf.FixedLenSequenceFeature(shape=[1],
                                                                         dtype=tf.int64,
                                                                         allow_missing=True),
            # '/dvs/camera_info/height':           tf.FixedLenSequenceFeature(shape=[1],
            #                                                                 dtype=tf.int64,
            #                                                                 allow_missing=True),
            # '/dvs/camera_info/width':            tf.FixedLenSequenceFeature(shape=[1],
            #                                                                 dtype=tf.int64,
            #                                                                 allow_missing=True),
            # '/dvs/camera_info/distortion':       tf.FixedLenFeature([], tf.string),
            # '/dvs/camera_info/D':                tf.FixedLenSequenceFeature(shape=[1],
            #                                                                 dtype=tf.float32,
            #                                                                 allow_missing=True),
            # '/dvs/camera_info/K':                tf.FixedLenSequenceFeature(shape=[1],
            #                                                                 dtype=tf.float32,
            #                                                                 allow_missing=True),
            # '/dvs/camera_info/R':                tf.FixedLenSequenceFeature(shape=[1],
            #                                                                 dtype=tf.float32,
            #                                                                 allow_missing=True),
            # '/dvs/camera_info/P':                tf.FixedLenSequenceFeature(shape=[1],
            #                                                                 dtype=tf.float32,
            #                                                                 allow_missing=True),
            # '/dvs/camera_info/binning_x':        tf.FixedLenSequenceFeature(shape=[1],
            #                                                                 dtype=tf.float32,
            #                                                                 allow_missing=True),
            # '/dvs/camera_info/binning_y':        tf.FixedLenSequenceFeature(shape=[1],
            #                                                                 dtype=tf.float32,
            #                                                                 allow_missing=True),
            # '/dvs/camera_info/roi':              tf.FixedLenSequenceFeature(shape=[1],
            #                                                                 dtype=tf.float32,
            #                                                                 allow_missing=True),
        }
        parsed_features = tf.parse_single_example(example, features=keys_to_features)
        consolidated_features = self._consolidate_features(parsed_features)

        data_instance = DataInstance()
        for topic, content in consolidated_features.items():
            if 'image_raw' in topic:
                image = tf.decode_raw(content['image'], tf.uint8)
                shape = tf.squeeze(tf.cast(content['shape'], tf.int32), [1])
                image = tf.reshape(image, shape)
                timestamp = tf.cast(content['timestamp'], tf.int32)
                image_record = ImageInstance(image, timestamp)
                data_instance[topic] = image_record

            # elif 'camera_info' in topic:
            #     height = tf.cast(content['height'], tf.int32)
            #     width = tf.cast(content['width'], tf.int32)
            #     distortion = tf.decode_raw(content['distortion'], tf.string)
            #     d = tf.cast(content['D'], tf.float32)
            #     k = tf.reshape(tf.cast(content['K'], tf.float32), shape=[3, 3])
            #     r = tf.reshape(tf.cast(content['R'], tf.float32), shape=[3, 3])
            #     p = tf.reshape(tf.cast(content['P'], tf.float32), shape=[3, 4])
            #     binning_x = tf.cast(content['binning_x'], tf.float32)
            #     binning_y = tf.cast(content['binning_y'], tf.float32)
            #     roi = tf.cast(content['roi'], tf.float32)
            #     camera_info = CameraInfo(height, width, distortion, d, k, r, p, binning_x,
            # binning_y, roi)
            #     data_instance[topic] = camera_info

            elif 'events' in topic:
                events = tf.decode_raw(content['encoded'], tf.int32)
                shape = tf.squeeze(tf.cast(content['shape'], tf.int32), [1])
                events = tf.reshape(events, shape)
                data_instance[topic] = events

            elif 'imu' in topic:
                imu_data = tf.decode_raw(content['encoded'], tf.int32)
                shape = tf.squeeze(tf.cast(content['shape'], tf.int32), [1])
                imu_data = tf.reshape(imu_data, shape)
                data_instance[topic] = imu_data

            elif 'objects' in topic:
                width = tf.cast(content['width'], tf.float32)
                height = tf.cast(content['height'], tf.float32)
                x_center = tf.cast(content['x_center'], tf.float32)
                y_center = tf.cast(content['y_center'], tf.float32)
                boxes = self._wrap_boxes(x_center, y_center, width, height)
                classes = tf.cast(content['classes'], tf.int32)
                scores = tf.cast(content['scores'], tf.float32)
                timestamp = tf.cast(content['timestamp'], tf.int32)
                object_detected = ObjectDetected(topic, boxes, classes, scores, timestamp)
                data_instance[topic] = object_detected

        return data_instance
"""
