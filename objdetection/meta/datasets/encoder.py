"""
author: az
"""
import io
import json
from abc import ABCMeta, abstractmethod

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


class Encoder(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def encode(self, instance):
        pass

    @abstractmethod
    def decode(self, instance):
        pass

    @staticmethod
    def _int64_feature(value):
        """Wrapper for inserting int64 features into Example proto."""
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def _float_feature(value):
        """Wrapper for inserting float features into Example proto."""
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @staticmethod
    def _bytes_feature(value):
        """Wrapper for inserting bytes features into Example proto."""
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    @staticmethod
    def _encode_image_array(image, format='PNG'):
        """
        Wrapper for encoding different types of image arrays.
        Following format options are available:
        'PNG'   use PIL to encode np.uint8 to PNG
        'BMP'   use PIL to encode np.uint8 to BMP
        '.png'  use CV2 to encode any kind to PNG
        '.bmp'  use CV2 to encode any kind to BMP
        :param image:
        :param format:
        :return:
        """
        if '.' in format:
            encoded_data = cv2.imencode(format, image)[1].tobytes()
        else:
            output_io = io.BytesIO()
            if format == 'BMP':
                # Workaround for PIL to be able to read 3-channel floating point arrays
                obj = image.astype(np.float32).tobytes()
                shape = image.shape
                mode = 'F'
                image_encoded_data = Image.frombuffer(mode, (shape[1], shape[0]), obj, "raw", mode,
                                                      0, 1)
                image_encoded_data.save(output_io, format=format)
            else:
                image_encoded_data = Image.fromarray(image)
                image_encoded_data.save(output_io, format=format)
            encoded_data = output_io.getvalue()
        return encoded_data

    @staticmethod
    def _center_to_minmax_boxes(boxes):
        """
        Convert object elements into boxes
        :param boxes: Nx4 [x_center, y_center, width, height]
        :return: [y_min, x_min, y_max, x_max]
        """

        if boxes.size != 0:
            xmin = boxes[:, 0] - boxes[:, 2] / 2
            ymin = boxes[:, 1] - boxes[:, 3] / 2
            xmax = boxes[:, 0] + boxes[:, 2] / 2
            ymax = boxes[:, 1] + boxes[:, 3] / 2
            return ymin, xmin, ymax, xmax
        else:
            return [np.empty(np.alen(boxes)) for i in range(4)]

    @staticmethod
    def _minmax_to_center_boxes(boxes):
        """
        Convert boxes from minmax to center coordinates
        including width and height
        :param boxes: Nx4 [y_min, x_min, y_max, x_max]
        :return: [x_center, y_center, width, height]
        """
        if boxes.size != 0:
            width = boxes[:, 3] - boxes[:, 1]  # x_max -x_min
            height = boxes[:, 2] - boxes[:, 0]  # y_max - y_min
            x_center = boxes[:, 1] + width / 2  # x_min + w
            y_center = boxes[:, 0] + height / 2  # y_min + w
            return x_center, y_center, width, height
        else:
            return [np.empty(np.alen(boxes)) for i in range(4)]

    @staticmethod
    def _load_json_labels(path_to_labels):
        with open(path_to_labels, "r") as f:
            raw_dict = json.load(f)
        # reformatting with key as int
        return {int(k): v for k, v in raw_dict.items()}

    @staticmethod
    def exportTFRecord(output_file, tf_record):
        try:
            with tf.python_io.TFRecordWriter(output_file) as writer:
                for example in tf_record:
                    writer.write(example.SerializeToString())
        except IOError:
            print('IOError attempting to serialize tfRecord file: %s' % output_file)
