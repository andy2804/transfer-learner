"""
author: aa & az
"""
import numpy as np

from objdetection.meta.observability.observer import MultiModalObserver
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
        :param labels:
        :param image:
        :param image_ts:
        :param events:
        :param boxes:
        :return:
        """
        labels_to_keep = np.empty((0,), dtype=np.int)
        boxes_to_keep = np.empty((0, 4), dtype=np.float)

        img_rgb, img_ir, boxes = list(map(np.copy, [img_rgb_in, img_ir_in, boxes]))
        img_rgb_grad = self._filter_img(img_rgb, filter='gradient')
        img_ir_grad = self._filter_img(img_ir, filter='gradient')
        for box_abs, class_id in zip(
                [self._box_norm_to_abs(box, img_rgb_grad) for box in boxes], labels):
            if class_id:
                img_grad_rgb_box = self._get_box_crop(img_rgb_grad, box_abs)
                img_grad_ir_box = self._get_box_crop(img_ir_grad, box_abs)
                perimeter = 2 * sum(img_grad_rgb_box.shape[:2])
                rgb_score = self._compute_observability_score([img_grad_rgb_box], type='rgb',
                                                              verbose=self._verbose)
                ir_score = self._compute_observability_score([img_grad_ir_box], type='rgb',
                                                             verbose=self._verbose)
                tl_keep = rgb_score > self._keep_thresh and ir_score > self._keep_thresh and \
                          perimeter >= self._min_img_perimeter
                if self.stats is not None:
                    self.stats.append_obj_stats(
                            label=class_id,
                            ymin=box_abs[0] / img_rgb.shape[0],
                            xmin=box_abs[1] / img_rgb.shape[1],
                            h=img_grad_rgb_box.shape[0] / img_rgb.shape[0],
                            w=img_grad_rgb_box.shape[1] / img_rgb.shape[1],
                            tl_score=(rgb_score if rgb_score < ir_score else ir_score),
                            tl_difficult=tl_keep)
                if tl_keep:
                    labels_to_keep = np.concatenate(([class_id], labels_to_keep), axis=0)
                    box_to_keep = self._abs_box_to_norm(box=box_abs, img=img_rgb_grad)
                    boxes_to_keep = np.concatenate(
                            ([box_to_keep], boxes_to_keep), axis=0)
        if self.stats is not None:
            self.stats.n_instances += 1
        return labels_to_keep, boxes_to_keep
