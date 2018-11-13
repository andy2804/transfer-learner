"""
author: az
"""
from abc import ABC, abstractmethod

import tensorflow as tf

from objdetection.meta.utils_generic.magic_constants import DAVIS240c
import numpy as np


class EventsTransform(ABC):
    """
    Class to transform an event tensor/np_array: [[ts,x,y,p],[ts,x,y,p],...] to frame
    """

    def __init__(self,
                 time_window=5,
                 h=DAVIS240c.height,
                 w=DAVIS240c.width):
        """

        :type time_window: float/int [ms]
        :param h:
        :param w:
        """
        self._time_span = time_window
        self._height = h
        self._width = w

    @abstractmethod
    def tf_transform_events(self, events, to_ts, keep_polarity):
        # fixme for the future implement from_ts
        pass

    @abstractmethod
    def np_transform_events(self, events, to_ts, keep_polarity):
        # fixme for the future implement from_ts
        pass

    @property
    def time_span(self):
        return self._time_span

    @time_span.setter
    def time_span(self, value):
        if value <= 0:
            raise ValueError("The time span to accumulate events cannot be negative or zero!")
        self._time_span = value

    @staticmethod
    def tf_crop_events_of_interest(events, from_ts, to_ts):
        """ Tensorflow
        :param events: [[ts,x,y,p],[ts,x,y,p],...]
        :param from_ts:
        :param to_ts:
        :return:
        """
        cond = tf.logical_and(tf.greater_equal(events[:, 0], from_ts),
                              tf.less_equal(events[:, 0], to_ts))
        return tf.gather_nd(events, tf.where(cond))

    @staticmethod
    def np_crop_events_of_interest(events, from_ts, to_ts):
        """ Numpy
        :param events: [[ts,x,y,p],[ts,x,y,p],...]
        :param from_ts:
        :param to_ts:
        :return:
        """
        cond = np.logical_and(np.greater_equal(events[:, 0], from_ts),
                              np.less_equal(events[:, 0], to_ts))
        return events[cond]
