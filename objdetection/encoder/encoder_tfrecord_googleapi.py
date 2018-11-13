import io

import numpy as np
import tensorflow as tf
from PIL import Image

from objdetection.encoder.encoder import Encoder


class EncoderTFrecGoogleApi(Encoder):

    def __init__(self, path_to_labels=None):
        super(EncoderTFrecGoogleApi, self).__init__()

        if path_to_labels is not None:
            labels_dict = self._load_json_labels(path_to_labels)

    def encode(self, instance, include_difficult_flag=False):
        """
        In this case the image is the RGB frame while the objects are gonna be inferred from
        the event based detection.
        Expected format boxes format: normalized and [ymin, xmin, ymax, xmax]
        :type instance: dict with keys image, boxes, labels
        :return:
        """
        # assert instance["image"].dtype is np.dtype(np.uint8)
        assert instance["image"].shape[2] == 3
        assert np.all(np.logical_and(np.less_equal(0, instance["boxes"]),
                                     np.less_equal(instance["boxes"], 1)))
        ymin, xmin, ymax, xmax = [instance["boxes"][:, i] for i in range(4)]
        assert np.all(np.logical_and(np.less_equal(ymin, ymax), np.less_equal(xmin, xmax)))

        # Encode image according to its value type
        if instance['image'].dtype is np.dtype(np.uint8):
            image_encoded = self._encode_image_array(instance['image'], format='PNG')
        else:
            image_encoded = self._encode_image_array(instance['image'], format='.png')
        image_format = b'png'
        image_shape = instance["image"].shape

        # Encode filename if available
        if 'filename' in instance:
            filename = instance['filename'].encode()
        else:
            filename = b'foo'

        # classes_text = [b'foo2'] * len(instance["labels"])  # todo adjust it!
        feature_map = {
            'image/height':             self._int64_feature(image_shape[0]),
            'image/width':              self._int64_feature(image_shape[1]),
            'image/filename':           self._bytes_feature(filename),
            'image/source_id':          self._bytes_feature(filename),
            'image/encoded':            self._bytes_feature(image_encoded),
            'image/format':             self._bytes_feature(image_format),
            'image/object/bbox/ymin':   self._float_feature(ymin.tolist()),
            'image/object/bbox/xmin':   self._float_feature(xmin.tolist()),
            'image/object/bbox/ymax':   self._float_feature(ymax.tolist()),
            'image/object/bbox/xmax':   self._float_feature(xmax.tolist()),
            # 'image/object/class/text':  self._bytes_feature(classes_text),
            'image/object/class/label': self._int64_feature(instance["labels"].tolist())
        }

        # Also include a list of difficult flags if demanded
        if include_difficult_flag:
            if 'difficult_flag' not in instance.keys():
                instance['difficult_flag'] = np.zeros(len(instance['labels']))
            feature_map['image/object/class/difficult'] = self._int64_feature(
                    instance["difficult_flag"].tolist())

        # Encode data to tf example
        tf_example = tf.train.Example(features=tf.train.Features(feature=feature_map))
        return tf_example

    def decode(self, instance):
        """
        Decodes tf example data into python lists. If include_difficult_flag is set to True,
        it will also try to decode difficult flag data. If it is not existent in the tfrecord
        dataset, the decoding process will likely crash.
        Returns dictionary with normalized bbox values
        :param instance:
        :param include_difficult_flag:
        :return:
        """
        keys_to_features = {
            'image/height':                 tf.FixedLenSequenceFeature(shape=[1], dtype=tf.int64,
                                                                       allow_missing=True),
            'image/width':                  tf.FixedLenSequenceFeature(shape=[1], dtype=tf.int64,
                                                                       allow_missing=True),
            'image/filename':               tf.FixedLenSequenceFeature([], tf.string,
                                                                       allow_missing=True),
            'image/source_id':              tf.FixedLenSequenceFeature([], tf.string,
                                                                       allow_missing=True),
            'image/encoded':                tf.FixedLenFeature([], tf.string),
            'image/format':                 tf.FixedLenFeature([], tf.string, default_value='raw'),

            # objects
            'image/object/bbox/ymin':       tf.FixedLenSequenceFeature(
                    shape=[1], dtype=tf.float32, allow_missing=True),
            'image/object/bbox/xmin':       tf.FixedLenSequenceFeature(
                    shape=[1], dtype=tf.float32, allow_missing=True),
            'image/object/bbox/ymax':       tf.FixedLenSequenceFeature(
                    shape=[1], dtype=tf.float32, allow_missing=True),
            'image/object/bbox/xmax':       tf.FixedLenSequenceFeature(
                    shape=[1], dtype=tf.float32, allow_missing=True),
            'image/object/class/label':     tf.FixedLenSequenceFeature(
                    shape=[1], dtype=tf.int64, allow_missing=True),
            'image/object/class/difficult': tf.FixedLenSequenceFeature(
                    shape=[1], dtype=tf.int64, allow_missing=True)
        }

        # Add difficult flag to the feature map for decoding
        # if include_difficult_flag:
        #     keys_to_features['image/object/class/difficult'] = tf.FixedLenSequenceFeature(
        #             shape=[1], dtype=tf.int64, allow_missing=True)

        # Decode features from tf example
        parsed_features = tf.parse_single_example(instance, features=keys_to_features)

        # instance id
        source_id = parsed_features['image/source_id']
        # im_format = parsed_features['image/format']
        # assert im_format == 'png'
        # parse image
        image = tf.image.decode_image(parsed_features['image/encoded'], channels=3)
        h = tf.squeeze(tf.cast(parsed_features['image/height'], tf.int32), [1])
        w = tf.squeeze(tf.cast(parsed_features['image/width'], tf.int32), [1])
        image = tf.reshape(image, tf.concat([h, w, [3]], axis=0))
        # parse labels & boxes

        ymin = tf.squeeze(parsed_features['image/object/bbox/ymin'], [1])
        xmin = tf.squeeze(parsed_features['image/object/bbox/xmin'], [1])
        ymax = tf.squeeze(parsed_features['image/object/bbox/ymax'], [1])
        xmax = tf.squeeze(parsed_features['image/object/bbox/xmax'], [1])
        classes = tf.squeeze(parsed_features['image/object/class/label'], [1])
        difficult_flag = tf.squeeze(parsed_features['image/object/class/difficult'], [1])
        # difficult_flag = tf.zeros(tf.shape(parsed_features['image/object/class/label']))
        return {'source_id':      source_id,
                'frame':          image,
                'ymin':           ymin,
                'xmin':           xmin,
                'ymax':           ymax,
                'xmax':           xmax,
                'gt_labels':      classes,
                'difficult_flag': difficult_flag
                }
