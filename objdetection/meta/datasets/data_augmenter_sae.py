"""
author: az
"""
import tensorflow as tf

from objdetection.rgb2events.nets import ssd_common
from objdetection.meta.datasets.data_augmenter import DataAugmenter
from objdetection.meta.utils_generic.magic_constants import DAVIS240c


# ====================== DATA AUGMENTATION
class DataAugSae(DataAugmenter):
    """ Class/container for data augmentation
    """

    def __init__(self, params=None):
        super().__init__(params)

        self._random_yshift_bool = self._dataaug_par.random_yshift_bool
        self._flipX_bool = self._dataaug_par.flipX_bool
        self._flipPolarity_bool = self._dataaug_par.flip_polarity_bool
        self._random_quant_bool = self._dataaug_par.random_quant_bool
        self._sample_distorted_bbox_bool = \
            self._dataaug_par.sample_distorted_bbox_bool
        self._sample_time_axis_bool = self._dataaug_par.sample_time_axis_bool
        self._sizeX = DAVIS240c.width
        self._sizeY = DAVIS240c.height

    def augment(self, input_proto):
        """ Builds the chain of data augmentation
        :param input_proto:{'frame':   image,
                'events':  events,
                'boxes':   boxes,
                'classes': classes,
                'scores':  scores}
        :returns : updates dictionary in place
        """
        # Local variables for better handling
        image, events = input_proto["frame"], input_proto["events"]
        xcent, ycent, w, h = input_proto["xcent"], input_proto["ycent"], input_proto["w"], \
                             input_proto["h"]
        # build chain of data aug
        if self._flipX_bool:
            image, events, xcent = self._flipX(image, events, xcent)
        if self._sample_distorted_bbox_bool:
            events, xcent, ycent, w, h = tf.cond(
                    tf.greater(tf.size(xcent), 0),
                    lambda: self._sample_distorted_bbox(events, xcent, ycent, w,
                                                        h),
                    lambda: (events, xcent, ycent, w, h))
        if self._sample_time_axis_bool:
            events = self._sample_time_axis(events)
        if self._random_quant_bool:
            events = self._random_quantization(events)
        if self._flipPolarity_bool:
            events = self._flip_polarity(events)
        if self._random_yshift_bool:
            events, ycent = tf.cond(
                    tf.greater(tf.reduce_sum(ycent), 0),
                    lambda: self._random_yshift(events, ycent, h),
                    lambda: (events, ycent))
        # update dict
        input_proto["image"], input_proto["events"] = image, events
        input_proto["xcent"], input_proto["ycent"], input_proto["w"], input_proto[
            "h"] = xcent, ycent, w, h

    @staticmethod
    def _flipX(image, events, xcent, prob=.5):
        """
        Randomly (with probability prob flips the X axis of the input)
        :return:
        """
        with tf.name_scope('random_flip_left_right'):
            uniform_random = tf.random_uniform([], 0, 1.0)
            mirror_cond = tf.less(uniform_random, prob)
            # Flip image.
            image = tf.cond(mirror_cond,
                            lambda: tf.image.flip_left_right(image),
                            lambda: image)
            events = tf.cond(mirror_cond,
                             lambda: tf.image.flip_left_right(events),
                             lambda: events)
            # Flip bboxes.
            xcent = tf.cond(mirror_cond,
                            lambda: 1.0 - xcent,
                            lambda: xcent)

            return image, events, xcent

    @staticmethod
    def _flip_polarity(events, prob=.5):
        """
        Flip polarity of events with probability prob.
        """
        with tf.name_scope('flip_polarity'):
            uniform_random = tf.random_uniform([], 0, 1.0)
            mirror_cond = tf.less(uniform_random, prob)
            events = tf.cond(mirror_cond,
                             lambda: tf.multiply(-1., events),
                             lambda: events)
            return events

    @staticmethod
    def _sample_distorted_bbox(events, xcent, ycent, w, h, prob=.5):
        """With probability prob crops a portion of the events image"""
        with tf.name_scope('sample_distorted_bbox'):
            uniform_random = tf.random_uniform([], 0, 1.0)
            mirror_cond = tf.less(uniform_random, prob)

            def sample_distortion(events_, xcent_, ycent_, w_, h_):
                xmin, ymin, xmax, ymax = \
                    ssd_common.bboxes_center_format_to_minmax(
                            xcent_, ycent_, w_, h_)
                bounding_boxes = tf.stack([ymin, xmin, ymax, xmax], axis=1)
                begin, size, bbox_distorted = \
                    tf.image.sample_distorted_bounding_box(
                            tf.shape(events_),
                            bounding_boxes=tf.expand_dims(bounding_boxes,
                                                          axis=0),
                            min_object_covered=0.95, area_range=[0.7, 1])
                events_ = tf.slice(events_, begin=begin, size=size)
                # todo generalize shape
                events_ = tf.image.resize_images(events_, size=[180, 240],
                                                 method=tf.image.ResizeMethod.BILINEAR)
                # Renormilize bboxes to deformed crop.
                ymin_crop, xmin_crop, ymax_crop, xmax_crop = tf.unstack(
                        tf.squeeze(bbox_distorted), axis=-1)
                xmin_new = tf.maximum(0., (xmin - xmin_crop) / (
                        xmax_crop - xmin_crop))
                ymin_new = tf.maximum(0., (ymin - ymin_crop) / (
                        ymax_crop - ymin_crop))
                xmax_new = 1 - tf.maximum(0., (xmax_crop - xmax) / (
                        xmax_crop - xmin_crop))
                ymax_new = 1 - tf.maximum(0., (ymax_crop - ymax) / (
                        ymax_crop - ymin_crop))
                # Reformat bboxes for distort events.
                xcent_, ycent_, w_, h_ = \
                    ssd_common.bboxes_minmax_format_to_center(
                            xmin_new, ymin_new, xmax_new, ymax_new)
                return events_, xcent_, ycent_, w_, h_

            return tf.cond(mirror_cond,
                           lambda: sample_distortion(events, xcent, ycent, w,
                                                     h),
                           lambda: (events, xcent, ycent, w, h))

    @staticmethod
    def _random_quantization(events, prob=.5):
        with tf.name_scope('random_quantization'):
            uniform_random = tf.random_uniform([], 0, 1.0)
            mirror_cond = tf.less(uniform_random, prob)

            def quantize_with_cast(events_):
                events_ = tf.cast(events_ * 255 / 2 + 255 / 2, dtype=tf.uint8)
                return tf.cast(events_, dtype=tf.float32) / (255. / 2.) - 1.

            return tf.cond(mirror_cond,
                           lambda: events,
                           lambda: quantize_with_cast(events))

    @staticmethod
    def _sample_time_axis(events, prob=1, remove_min_percent=0,
                          remove_max_percent=40):
        """ Expect as inputs events scaled between -1 and 1, randomly we 
        remove between 0 and
        max_percetage_remove of the oldest events (values closer to zero)
        :param events:
        :param prob:
        :return:
        """
        with tf.name_scope('random_time_axis_sampling'):
            uniform_random = tf.random_uniform([], 0, 1.0)
            mirror_cond = tf.less(uniform_random, prob)

            def crop_oldest_events_and_rescale(events_):
                percentage_tocrop = tf.random_uniform([], remove_min_percent,
                                                      remove_max_percent)
                thresh_tocrop = percentage_tocrop / 100.0
                # cond returns true for events in [- thresh_tocrop, 
                # thresh_tocrop]
                cond = tf.logical_and(tf.greater(thresh_tocrop, events_),
                                      tf.greater(events_, - thresh_tocrop))
                events_ = tf.where(cond, tf.zeros_like(events_), events_)
                # rescaling of non zeros between 0 and 1 or -1
                m = 1 / (1 - thresh_tocrop)
                rescaled_pos = m * events_ + (1 - m)
                rescaled_neg = m * events_ + (m - 1)
                events_ = tf.where(tf.greater(events_, thresh_tocrop),
                                   rescaled_pos, events_)
                events_ = tf.where(tf.greater(- thresh_tocrop, events_),
                                   rescaled_neg, events_)
                return events_

            return tf.cond(mirror_cond,
                           lambda: events,
                           lambda: crop_oldest_events_and_rescale(events))

    @staticmethod
    def _random_yshift(events, ycent, h, prob=.4):
        # todo to test
        with tf.name_scope('random_yshift'):
            uniform_random = tf.random_uniform([], 0, 1.0)
            mirror_cond = tf.less(uniform_random, prob)

            def internal_yshift(_image, _ycoord, _height):
                ymin_raw = _ycoord - _height / 2
                ymin_filt = tf.gather(ymin_raw,
                                      tf.where(tf.not_equal(_height, 0)))
                ymin_bboxes = tf.reduce_min(ymin_filt)
                maxval = tf.maximum(1, tf.cast(tf.reduce_min(ymin_bboxes * 180),
                                               dtype=tf.int32))
                yshift = tf.random_uniform([1], minval=0, maxval=maxval,
                                           dtype=tf.int32)
                _image = tf.slice(_image, tf.concat([yshift, [0, 0]], axis=0),
                                  [-1, -1, -1])
                _image = tf.image.pad_to_bounding_box(_image, offset_height=0,
                                                      offset_width=0,
                                                      target_height=180,
                                                      target_width=240)
                shifted = _ycoord - tf.cast(yshift, dtype=tf.float32) / 180
                _ycoord = tf.where(tf.not_equal(_ycoord, 0), shifted, _ycoord)
                return _image, _ycoord

        return tf.cond(mirror_cond,
                       lambda: (events, ycent),
                       lambda: internal_yshift(events, ycent, h))
