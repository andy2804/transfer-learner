"""
author: az
"""
from abc import ABC, abstractmethod

import tensorflow as tf

from objdetection.meta.utils_generic import magic_constants


class ObjectDetector(ABC):
    """
    """

    def __init__(self, parameters):
        self._objdet_par = None, None
        self.objdet_par = parameters
        self.is_training = tf.placeholder(tf.bool, name='is_training')

    @abstractmethod
    def parse_input(self, input_proto):
        pass

    @abstractmethod
    def forwardpass(self):
        pass

    @abstractmethod
    def tower_losses(self):
        pass

    @abstractmethod
    def optimizer(self):
        pass

    """@abstractmethod
    def nms(self):
        pass
    """

    @property
    def objdet_par(self):
        return self._objdet_par

    @objdet_par.setter
    def objdet_par(self, parameters):
        if isinstance(parameters, magic_constants.ObjDetParams):
            self._objdet_par = parameters
        else:
            raise ValueError('Parameters value is not acceptable')


if __name__ == '__main__':
    pass
    # print("Testing only!!")
    # args = {"is_training": True,
    #         "options":     magic_constants.DEFAULT_OPTIONS,
    #         "params":      magic_constants.DEFAULT_PARAMS
    #         }
    # obj_detector = ObjectDetector(**args)
    # print("test ended")
