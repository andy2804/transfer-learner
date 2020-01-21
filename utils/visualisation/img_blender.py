from collections import OrderedDict

import cv2
import numpy as np

from objdetection.meta.utils_labeler.static_helper import load_labels
from objdetection.meta.visualisation.static_helper import add_text_overlay


class LFImageBlender:
    def __init__(self, labels='zauron_label_map.json'):
        self._labels = load_labels(labels)
        self._avg_look = OrderedDict().fromkeys(
                list(self._labels.keys()), (0, np.zeros([180, 240, 3], dtype=np.float64)))

    def _update_avg_img(self, img, class_id, decay=False):
        '''
        Updates the avg image for the input image class in an exponential decay way
        :param img:
        :param class_id:
        :param decay:
        :return:
        '''
        count = self._avg_look[class_id][0]
        frame = self._avg_look[class_id][1]
        count += 1
        if decay:
            frame = (frame * np.exp(-1.0 / 10.0)) + img
        else:
            frame = frame + img
        self._avg_look[class_id] = (count, frame)
        self._create_avg_overview()

    def _create_avg_overview(self, rows=2):
        '''
        Reads out the images from the dictionary and creates a verbose overview
        :param rows:
        :return:
        '''
        cols = int(np.ceil(len(self._avg_look) / rows))
        v_stack = []
        total_count = 0
        for r in range(rows):
            h_stack = []
            for c in range(cols):
                class_id = r * cols + c + 1
                if class_id in self._avg_look:
                    count = self._avg_look[class_id][0]
                    total_count += count
                    frame = self._avg_look[class_id][1]
                    output = np.zeros(frame.shape, dtype=np.uint8)
                    if count > 0:
                        cv2.normalize(frame, output, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                      dtype=cv2.CV_8U)
                    output = add_text_overlay(output,
                                              '%s: %i' % (self._labels[class_id]['name'], count),
                                              overlay=False)
                    h_stack.append(output)
                else:
                    h_stack.append(np.zeros(output.shape, dtype=np.uint8))
            v_stack.append(np.hstack(h_stack))
        overview = np.vstack(v_stack)
        cv2.imshow('Average learned objects', overview)
