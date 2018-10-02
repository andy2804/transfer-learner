"""
author: aa & az
"""
import numpy as np

from objdetection.meta.observability.observer import MultiModalObserver
from objdetection.meta.visualisation.static_helper import (draw_img_from_events)
from objdetection.rgb2events.tfrecords_builder.stats import TLStatistician


class TFRecordLearningFilter(MultiModalObserver):
    def __init__(self, events_time_window=33, score_threshold=13, min_img_perimeter=80,
                 logstats=False):
        MultiModalObserver.__init__(self)
        self.stats = TLStatistician(tl_score_threshold=score_threshold) if logstats else None
        self._keep_thresh = score_threshold
        self._min_img_perimeter = min_img_perimeter
        self._events_time_window = events_time_window

    def apply_to_tfrec(self, image, image_ts, events, labels, boxes):
        """
        Learning filter to be applied in tfrecords converter
        :param labels:w
        :param image:
        :param image_ts:
        :param events:
        :param boxes:
        :return:
        """
        labels_to_keep = np.empty((0,), dtype=np.int)
        boxes_to_keep = np.empty((0, 4), dtype=np.float)

        image, events, boxes = list(map(np.copy, [image, events, boxes]))
        img_gradient = self._filter_img(image, what_to_compute='gradient')
        events = self._get_events_until_ts(
                events, image_ts, time_window=self._events_time_window)
        events_img = draw_img_from_events(events, shape=image.shape[0:2])
        for box_abs, class_id in zip(
                [self._box_norm_to_abs(box, img_gradient) for box in boxes], labels):
            if class_id:
                img_box = self._get_box_crop(img_gradient, box_abs)
                perimeter = 2 * sum(img_box.shape[:2])
                events_img_box = self._get_box_crop(events_img, box_abs)
                tl_score = self._compute_observability_score([img_box, events_img_box],
                                                             type='events')
                tl_keep = tl_score > self._keep_thresh and perimeter >= self._min_img_perimeter
                if self.stats is not None:
                    self.stats.append_obj_stats(
                            label=class_id,
                            ymin=box_abs[0] / image.shape[0],
                            xmin=box_abs[1] / image.shape[1],
                            h=img_box.shape[0] / image.shape[0],
                            w=img_box.shape[1] / image.shape[1],
                            tl_score=tl_score,
                            tl_difficult=tl_keep)
                if tl_keep:
                    labels_to_keep = np.concatenate(([class_id], labels_to_keep), axis=0)
                    box_to_keep = self._abs_box_to_norm(box=box_abs, img=img_gradient)
                    boxes_to_keep = np.concatenate(
                            ([box_to_keep], boxes_to_keep), axis=0)
        if self.stats is not None:
            self.stats.n_instances += 1
        return labels_to_keep, boxes_to_keep
