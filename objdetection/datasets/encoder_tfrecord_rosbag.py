"""
author: az
"""
import os

import numpy as np
import tensorflow as tf

from objdetection.meta.datasets.encoder import Encoder
from rosbag_converter.data_instance import (ImageInstance, ObjectDetected)


class EncoderTFrecRosbag(Encoder):
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

    def encode(self, instance, rosbag_name):
        """
        Build example proto from DataInstance
        :param instance:
        :return: tf.train.Example
        """
        unique_id = '%s_%d' % (rosbag_name, instance['/pylon_rgb/image_raw'].ts.to_nsec())
        # No specific encoding results in 'utf-8' which is the fastest
        features = {'source_id': self._bytes_feature(unique_id.encode())}

        for key, value in instance.items():
            if isinstance(value, ImageInstance) and 'pylon_rgb' in key:
                image_format = b'raw'
                features['/pylon_rgb/image_raw/image'] = self._bytes_feature(
                        value.image.tostring())
                features['/pylon_rgb/image_raw/format'] = self._bytes_feature(
                        image_format)
                features['/pylon_rgb/image_raw/shape'] = self._int64_feature(
                        list(value.image.shape))
                features['/pylon_rgb/image_raw/ts'] = self._int64_feature(
                        value.ts.to_nsec())

            elif value.__class__.__name__ == 'ndarray':
                if 'events' in key:
                    features['/dvs/events/ts'] = self._int64_feature(
                            value[:, 0].tolist())
                    features['/dvs/events/xyp'] = self._bytes_feature(
                            (value[:, 1:].astype(np.uint8)).tostring())
                    features['/dvs/events/xyp_shape'] = self._int64_feature(
                            list(value[:, 1:].shape))
                elif 'imu' in key:
                    pass
                else:
                    pass

            elif isinstance(value, ObjectDetected):
                x_center, y_center, width, height = self._minmax_to_center_boxes(value.boxes)
                features['/pylon_rgb/image_raw/objects/x_center'] = self._float_feature(
                        x_center.tolist())
                features['/pylon_rgb/image_raw/objects/y_center'] = self._float_feature(
                        y_center.tolist())
                features['/pylon_rgb/image_raw/objects/width'] = self._float_feature(
                        width.tolist())
                features['/pylon_rgb/image_raw/objects/height'] = self._float_feature(
                        height.tolist())
                features['/pylon_rgb/image_raw/objects/classes'] = self._int64_feature(
                        value.classes.tolist())
                features['/pylon_rgb/image_raw/objects/scores'] = self._float_feature(
                        value.scores.tolist())
                features['/pylon_rgb/image_raw/objects/ts'] = self._int64_feature(
                        value.ts.to_nsec())

            else:
                pass
        example = tf.train.Example(features=tf.train.Features(feature=features))
        return example

    def decode(self, instance):
        """
        :param instance:
        :return:
        """
        keys_to_features = {
            'source_id':                             tf.FixedLenSequenceFeature(
                    [], tf.string, allow_missing=True),
            '/pylon_rgb/image_raw/image':            tf.FixedLenFeature(
                    [], tf.string),
            '/pylon_rgb/image_raw/format':           tf.FixedLenFeature(
                    [], tf.string, default_value='raw'),
            '/pylon_rgb/image_raw/shape':            tf.FixedLenSequenceFeature(
                    shape=[1], dtype=tf.int64, allow_missing=True),
            '/pylon_rgb/image_raw/ts':               tf.FixedLenSequenceFeature(
                    shape=[1], dtype=tf.int64, allow_missing=True),
            # objects
            '/pylon_rgb/image_raw/objects/x_center': tf.FixedLenSequenceFeature(
                    shape=[1], dtype=tf.float32, allow_missing=True),
            '/pylon_rgb/image_raw/objects/y_center': tf.FixedLenSequenceFeature(
                    shape=[1], dtype=tf.float32, allow_missing=True),
            '/pylon_rgb/image_raw/objects/width':    tf.FixedLenSequenceFeature(
                    shape=[1], dtype=tf.float32, allow_missing=True),
            '/pylon_rgb/image_raw/objects/height':   tf.FixedLenSequenceFeature(
                    shape=[1], dtype=tf.float32, allow_missing=True),
            '/pylon_rgb/image_raw/objects/classes':  tf.FixedLenSequenceFeature(
                    shape=[1], dtype=tf.int64, allow_missing=True),
            '/pylon_rgb/image_raw/objects/scores':   tf.FixedLenSequenceFeature(
                    shape=[1], dtype=tf.float32, allow_missing=True),
            '/pylon_rgb/image_raw/objects/ts':       tf.FixedLenSequenceFeature(
                    shape=[1], dtype=tf.int64, allow_missing=True),
            # events
            '/dvs/events/ts':                        tf.FixedLenSequenceFeature(
                    shape=[1], dtype=tf.int64, allow_missing=True),
            '/dvs/events/xyp':                       tf.FixedLenFeature([], tf.string),
            '/dvs/events/xyp_shape':                 tf.FixedLenSequenceFeature(
                    shape=[1], dtype=tf.int64, allow_missing=True),
        }

        parsed_features = tf.parse_single_example(instance, features=keys_to_features)
        # instance id
        source_id = parsed_features['source_id']
        # parse image
        image = tf.decode_raw(parsed_features['/pylon_rgb/image_raw/image'], tf.uint8)
        shape_im = tf.squeeze(
                tf.cast(parsed_features['/pylon_rgb/image_raw/shape'], tf.int32), [1])
        image = tf.reshape(image, shape_im)
        image_ts = tf.squeeze(
                tf.cast(parsed_features['/pylon_rgb/image_raw/ts'], dtype=tf.int64), axis=-1)
        # parse events
        events_ts = tf.cast(parsed_features['/dvs/events/ts'], dtype=tf.float64)
        events_xyp = tf.decode_raw(parsed_features['/dvs/events/xyp'], tf.uint8)
        shape_ev = tf.squeeze(
                tf.cast(parsed_features['/dvs/events/xyp_shape'], tf.int32), [1])
        events_xyp = tf.cast(tf.reshape(events_xyp, shape_ev), dtype=tf.float64)
        events = tf.concat([events_ts, events_xyp], axis=1)
        # parse labels & boxes
        xcent = tf.squeeze(parsed_features['/pylon_rgb/image_raw/objects/x_center'], [1])
        ycent = tf.squeeze(parsed_features['/pylon_rgb/image_raw/objects/y_center'], [1])
        w = tf.squeeze(parsed_features['/pylon_rgb/image_raw/objects/width'], [1])
        h = tf.squeeze(parsed_features['/pylon_rgb/image_raw/objects/height'], [1])
        classes = tf.squeeze(parsed_features['/pylon_rgb/image_raw/objects/classes'], [1])
        scores = tf.squeeze(parsed_features['/pylon_rgb/image_raw/objects/scores'], [1])
        return {'source_id': source_id,
                'frame':     image,
                'frame_ts':  image_ts,
                'events':    events,
                'xcent':     xcent,
                'ycent':     ycent,
                'w':         w,
                'h':         h,
                'gt_labels': classes,
                'scores':    scores
                }
