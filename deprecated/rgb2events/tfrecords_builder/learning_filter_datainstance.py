"""
author: aa & az
"""

import cv2
import numpy as np

from objdetection.meta.observability.observer import MultiModalObserver
from objdetection.meta.visual.img_blender import LFImageBlender
from objdetection.meta.visual.static_helper import (draw_img_from_events,
                                                    draw_overlay_from_events,
                                                    visualize_rgb_detections)
from objdetection.rgb2events.tfrecords_builder.learning_tracker import LearningTracker


# =============================================
# Hand-labeling key inputs:
# ENTER:    Learn Event Frame
# ESC:      Discard Event Frame
# X:        Reverse Last Assignment
# =============================================

class DatainstanceLearningFilter(MultiModalObserver):
    def __init__(self, events_time_window=33, keep_threshold=13, min_img_perimeter=80):
        MultiModalObserver.__init__(self)
        self._keep_thresh = keep_threshold
        self._min_img_perimeter = min_img_perimeter
        self._events_time_window = events_time_window
        self.lf_tracker = LearningTracker()
        self.hand_tracker = LearningTracker()
        self._img_blender = LFImageBlender()

    def apply_to_datainstance(self, datainstance, img_topic='/pylon_rgb/image_raw', verbose=False,
                              hand_label=False):
        """
        Learning filter to be applied during rosbag conversion
        :param datainstance:
        :param img_topic:
        :param verbose:
        :return:
        """
        labels_to_keep = np.zeros((1,), dtype=np.int)
        boxes_to_keep = np.zeros((1, 4), dtype=np.float)
        object_topic = img_topic + '/objects'

        # Get Original img
        img_rgb = cv2.cvtColor(np.copy(datainstance[img_topic].image), cv2.COLOR_BGR2RGB)

        # Get gradient img & events
        shape = (img_rgb.shape[0], img_rgb.shape[1])
        img_gradient = self._filter_img(img_rgb, what_to_compute='gradient')
        events = datainstance['/dvs/events']
        ts = datainstance[img_topic].ts.to_nsec()
        event_window = self._get_events_until_ts(events, ts, time_window=self._events_time_window)
        img_events_window = draw_img_from_events(event_window, shape)

        # Check each box
        box_count = 0
        for box_abs, class_id in zip([self._box_norm_to_abs(box, img_gradient) for box in
                                      datainstance[object_topic].boxes],
                                     datainstance[object_topic].classes):

            img_box = self._get_box_crop(img_gradient, box_abs)
            img_rgb_box = self._get_box_crop(img_rgb, box_abs)
            entropy = self._compute_observability_score([img_rgb_box], type='rgb', verbose=verbose)
            perimeter = 2 * sum(img_box.shape[:2])
            events_img_box = self._get_box_crop(img_events_window, box_abs)
            tl_score = self._compute_observability_score([img_box, events_img_box], type='events')
            tl_keep = tl_score > self._keep_thresh and perimeter >= self._min_img_perimeter
            if tl_keep:
                # print('[ %i%% >= %i%% ] ACCEPTING:\t%s' % (tl_score, self._keep_thresh,
                #                                            self._labels[class_id]['name']))
                labels_to_keep = np.concatenate(([class_id], labels_to_keep), axis=0)
                box_to_keep = self._abs_box_to_norm(box=box_abs, img=img_gradient)
                boxes_to_keep = np.concatenate(
                        ([box_to_keep], boxes_to_keep), axis=0)
                tl_keep = True

            # Debug image stacked
            if verbose or hand_label:
                img_rgb_labelled = visualize_rgb_detections(img_rgb, datainstance[object_topic],
                                                            labels=self._labels)
                img_events_overlay = draw_overlay_from_events(self._events_time_window,
                                                              img_rgb_labelled,
                                                              max_images=1)
                key = self._create_verbose_image(tl_score, tl_keep, class_id, box_abs, hand_label,
                                                 img_box, events_img_box, img_rgb_labelled,
                                                 img_events_overlay, img_gradient,
                                                 img_events_window)
                while key is not None:
                    if key == 13:
                        object = self.hand_tracker.add(True, tl_score, class_id, box_abs, ts)
                        print('\nLast object %s assignment set to True!' % object)
                        break
                    elif key == 27:
                        object = self.hand_tracker.add(False, tl_score, class_id, box_abs, ts)
                        print('\nLast object %s assignment set to False!' % object)
                        break
                    elif key == 120:
                        object, value = self.hand_tracker.pop()
                        value['learn'] = not value['learn']
                        self.hand_tracker[object] = value
                        self.hand_tracker.change_counter(value['object'], 2 * value['learn'] - 1)
                        print('\nLast object %s assignment changed from %s to %s!' % (
                            object, not value['learn'], value['learn']))
                        self.hand_tracker.pretty_print()
                        self.lf_tracker.pretty_print()
                        key = cv2.waitKey(0)
                    else:
                        print('\nInvalid Key!')
                        key = cv2.waitKey(0)

            if hand_label:
                self.hand_tracker.pretty_print()
                self.lf_tracker.add(tl_keep, tl_score, class_id, box_abs, ts)
                self.lf_tracker.pretty_print()
            box_count += 1
        return labels_to_keep, boxes_to_keep
