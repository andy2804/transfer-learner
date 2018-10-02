import random

import numpy as np


class DetectionFilter:
    def __init__(self, label_map_id, min_numb_of_objs=1,
                 keep_thresh=.5, motion_thresh=0.4,
                 long_record_filter=None, p_retain_empty=.5,
                 for_testing=False):
        """
        :param label_map_id:
        :param min_numb_of_objs:
        :param keep_thresh:
        :param motion_thresh:
        :param long_record_filter:
        """

        if long_record_filter is None:
            long_record_filter = []
        self._label_map_id = label_map_id
        self._min_numb_of_objs = min_numb_of_objs
        self._keep_thresh = keep_thresh
        self._for_testing = for_testing
        self._motion_thresh = motion_thresh
        self._long_record_filter = long_record_filter
        self._p_retain_empty = p_retain_empty

    def process_rawinference(self, detected):
        success = False
        bool2keep = np.ones_like(detected['scores'])
        bool2keep[np.where(detected['scores'] < self.keep_thresh)] = 0
        half_success = np.greater_equal(np.sum(bool2keep),
                                        self._min_numb_of_objs)
        new_detected = {"events":  detected["events"],
                        "image":   detected["image"],
                        "classes": [], "scores": [], "boxes": []
                        }
        # Half success if we found at least min_numb_of_objs with score > 
        # keep_thresh
        if half_success:
            idx_to_map = np.argwhere(bool2keep > 0)
            for idx in idx_to_map:
                old_class_id = detected['classes'][idx][0] if \
                    detected['classes'][idx] < 10 else 0
                new_class_id = self._label_map_id[old_class_id]
                # If new class id not zero (background class)
                if new_class_id and self._is_there_motion(detected["events"],
                                                          detected['boxes'][
                                                              idx][0]):
                    new_detected["classes"].append(new_class_id)
                    new_detected["scores"].append(detected['scores'][idx][0])
                    new_detected["boxes"].append(detected['boxes'][idx][0])
                    success = True
            # filter out from long recordings too popular objects
            if new_detected["classes"]:
                success = self._filter_too_popular(new_detected["classes"])
            # if we want to return also negative examples without any object
            elif self.min_num_of_objs == 0:
                success = True if self.p_retain_empty < random.random() or \
                                  self._for_testing else False
        return new_detected, success

    def _is_there_motion(self, events, box):
        """Empirical check to make sure that we have some events where we 
        found an objects."""
        if self._for_testing:
            return True
        else:
            box = np.concatenate([events.shape] * 2) * box
            # Compute perimeter of the box (ymin, xmin, ymax, xmax)
            peri = ((box[2] - box[0]) + (box[3] - box[1])) * 2
            box_i = np.round(box).astype(np.int64)
            events = np.abs(events)
            cropped_box = events[box_i[0]:box_i[2], box_i[1]:box_i[3]]
            activity_measure = np.sum(cropped_box) / peri
            return True if activity_measure > self.motion_thresh else False

    def _filter_too_popular(self, detected_classes):
        """Return false if the only detected object are those too popular.
        True if at least one detected is dot in the list of too popular"""
        return any(d not in self.long_record_filter for d in detected_classes)

    # ==========
    @property
    def keep_thresh(self, ):
        return self._keep_thresh

    @keep_thresh.setter
    def keep_thresh(self, value):
        if value < 0 or value > 1:
            raise ValueError(
                    "Keep threshold variable must be within 0 and 1. It's a "
                    "score "
                    "confidence.")
        self._keep_thresh = value

    @property
    def min_num_of_objs(self, ):
        return self._min_numb_of_objs

    @min_num_of_objs.setter
    def min_num_of_objs(self, value):
        self._min_numb_of_objs = value

    @property
    def motion_thresh(self, ):
        return self._motion_thresh

    @motion_thresh.setter
    def motion_thresh(self, value):
        self._motion_thresh = value

    @property
    def long_record_filter(self, ):
        return self._long_record_filter

    @long_record_filter.setter
    def long_record_filter(self, value):
        if not isinstance(value, list):
            raise ValueError("The filter has to be a list of integers")
        self._long_record_filter = value

    @property
    def p_retain_empty(self, ):
        return self._p_retain_empty

    @p_retain_empty.setter
    def p_retain_empty(self, value):
        if value < 0 or value > 1:
            raise ValueError(
                    "Probability of retaining an empty must be within 0 and 1.")
        self._p_retain_empty = value
