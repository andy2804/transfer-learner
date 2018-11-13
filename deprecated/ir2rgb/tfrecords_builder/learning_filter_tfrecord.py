"""
author: aa & az
"""

import numpy as np

from matplotlib import pyplot as plt
from objdetection.meta.observability.observer import MultiModalObserver
from objdetection.meta.visual.static_helper import add_text_overlay
from objdetection.rgb2events.tfrecords_builder.stats import TLStatistician


class TFRecordLearningFilter(MultiModalObserver):
    def __init__(self, score_threshold=0, min_img_perimeter=80, logstats=False, verbose=False):
        MultiModalObserver.__init__(self)
        self.stats = TLStatistician(tl_score_threshold=score_threshold) if logstats else None
        self._keep_thresh = score_threshold
        self._min_img_perimeter = min_img_perimeter
        self._verbose = verbose

    def apply_to_tfrec(self, img_rgb_in, img_ir_in, labels, boxes):
        """
        Learning filter to be applied in tfrecords converter. Calculates the shannon entropy
        of objects detected on input images. If the object is visible in both domains (meaning
        that the score is above the defined threshold), the image object pair is kept.
        :param labels:w
        :param image:
        :param image_ts:
        :param events:
        :param boxes:
        :return:
        """
        # labels_to_keep = np.empty((0,), dtype=np.int)
        # boxes_to_keep = np.empty((0, 4), dtype=np.float)
        difficult = []

        img_rgb, img_ir, boxes = list(map(np.copy, [img_rgb_in, img_ir_in, boxes]))
        img_rgb_filtered = self._filter_img(img_rgb, filter='grayscale', normalize=False)
        img_ir_filtered = self._filter_img(img_ir, filter='grayscale', normalize=False)

        for box_abs, class_id in zip(
                [self._box_norm_to_abs(box, img_rgb_filtered) for box in boxes], labels):
            if class_id:
                img_rgb_box = self._get_box_crop(img_rgb_filtered, box_abs)
                img_ir_box = self._get_box_crop(img_ir_filtered, box_abs)
                perimeter = 2 * sum(img_rgb_box.shape[:2])
                rgb_score = self._compute_observability_score([img_rgb_box], type='rgb',
                                                              verbose=False)
                ir_score = self._compute_observability_score([img_ir_box], type='rgb',
                                                             verbose=False)
                if self._verbose:
                    img_rgb_box = add_text_overlay(img_rgb_box, '%.2f' % rgb_score, overlay=False)
                    img_ir_box = add_text_overlay(img_ir_box, '%.2f' % ir_score, overlay=False)
                    img_stack = np.hstack((img_rgb_box, img_ir_box))
                    plt.imshow(img_stack, cmap='gray')
                    plt.show()

                tl_keep = rgb_score > self._keep_thresh and perimeter >= self._min_img_perimeter
                if self.stats is not None:
                    self.stats.append_obj_stats(
                            label=class_id,
                            ymin=box_abs[0] / img_rgb.shape[0],
                            xmin=box_abs[1] / img_rgb.shape[1],
                            h=img_rgb_box.shape[0] / img_rgb.shape[0],
                            w=img_rgb_box.shape[1] / img_rgb.shape[1],
                            tl_score=rgb_score,
                            tl_difficult=tl_keep)
                    difficult.append(0) if tl_keep else difficult.append(1)
                    # labels_to_keep = np.concatenate(([class_id], labels_to_keep), axis=0)
                    # box_to_keep = self._abs_box_to_norm(box=box_abs, img=img_rgb_filtered)
                    # boxes_to_keep = np.concatenate(
                    #         ([box_to_keep], boxes_to_keep), axis=0)

        if self.stats is not None:
            self.stats.n_instances += 1
        return np.array(difficult)
