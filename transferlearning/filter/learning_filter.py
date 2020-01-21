"""
author: aa & az
"""
import numpy as np

from objdetection.observability.observer import MultiModalObserver
from transferlearning.filter.stats import TLStatistician


class LearningFilter(MultiModalObserver):
    def __init__(self, score_threshold=0, min_img_perimeter=80, logstats=False, mode='rgb',
                 verbose=False):
        MultiModalObserver.__init__(self)
        self.stats = TLStatistician(tl_score_threshold=score_threshold) if logstats else None
        self._keep_thresh = score_threshold
        self._min_img_perimeter = min_img_perimeter
        self._observability_mode = mode
        self._verbose = verbose

    def apply(self, img_main_in, img_aux_in, classes, boxes):
        """
        Learning filter to be applied in tfrecords converter. Calculates the shannon entropy
        of objects detected on input images. If the object is visible in both domains (meaning
        that the score is above the defined threshold), the image object pair is kept.
        :param classes:
        :param image:
        :param image_ts:
        :param events:
        :param boxes:
        :return:
        """
        classes_to_keep = np.empty((0,), dtype=np.int)
        boxes_to_keep = np.empty((0, 4), dtype=np.float)

        img_main, img_aux, boxes = list(map(np.copy, [img_main_in, img_aux_in, boxes]))

        if self._observability_mode == 'rgb':
            img_main_grad = self._filter_img(img_main, filter='grayscale', normalize=True,
                                             mode='events')
            img_aux_grad = self._filter_img(img_aux, filter='gradient', normalize=True,
                                            mode=self._observability_mode)
        else:
            img_main_grad = self._filter_img(img_main, filter='gradient', normalize=True, mode='rgb')
            img_aux_grad = self._filter_img(img_aux, filter='grayscale', normalize=True,
                                            mode=self._observability_mode)
        for box_abs, class_id in zip(
                [self._box_norm_to_abs(box, img_main_grad) for box in boxes], classes):
            if class_id:
                img_main_grad_box = self._get_box_crop(img_main_grad, box_abs)
                img_aux_grad_box = self._get_box_crop(img_aux_grad, box_abs)
                perimeter = 2 * sum(img_main_grad_box.shape[:2])
                aux_sensor_score = self._compute_observability_score(
                        [img_main_grad_box, img_aux_grad_box],
                        type=self._observability_mode,
                        verbose=self._verbose)
                tl_keep = aux_sensor_score >= self._keep_thresh and perimeter >= \
                          self._min_img_perimeter

                # Record statistics
                if self.stats is not None:
                    self.stats.append_obj_stats(
                            label=class_id,
                            ymin=box_abs[0] / img_main.shape[0],
                            xmin=box_abs[1] / img_main.shape[1],
                            h=img_main_grad_box.shape[0] / img_main.shape[0],
                            w=img_main_grad_box.shape[1] / img_main.shape[1],
                            tl_score=aux_sensor_score,
                            tl_difficult=tl_keep)
                if tl_keep:
                    classes_to_keep = np.concatenate(([class_id], classes_to_keep), axis=0)
                    box_to_keep = self._abs_box_to_norm(box=box_abs, img=img_main_grad)
                    boxes_to_keep = np.concatenate(([box_to_keep], boxes_to_keep), axis=0)

        if self.stats is not None:
            self.stats.n_instances += 1
        return classes_to_keep, boxes_to_keep

    def remove_boxes_from_roi(self, classes, boxes, roi=None, shape=None, tolerance=0.15):
        """
        If roi is defined, boxes with given shape lying within the roi will be removed
        :param classes:
        :param boxes:
        :param roi: Tuple of float specifying the corner points similar to boxes
                    with (y_min, x_min, y_max, x_max)
        :param shape: Tuple of float specifying height and width
        :param tolerance: Tolerance of shape
        :return:
        """
        if roi is not None:
            classes_to_keep = np.empty((0,), dtype=np.int)
            boxes_to_keep = np.empty((0, 4), dtype=np.float)
            for box, class_id in zip(boxes, classes):
                if not self.check_box_in_roi(box, roi) and not self.check_box_shape(box, shape,
                                                                                    tolerance):
                    classes_to_keep = np.concatenate(([class_id], classes_to_keep), axis=0)
                    boxes_to_keep = np.concatenate(([box], boxes_to_keep), axis=0)
            return classes_to_keep, boxes_to_keep
        return classes, boxes

    @staticmethod
    def check_box_in_roi(box, roi):
        if box[0] >= roi[0] and box[1] >= roi[1] and box[2] <= roi[2] and box[3] <= roi[3]:
            return True
        return False

    @staticmethod
    def check_box_shape(box, shape, tolerance):
        if abs(1.0 - ((box[2] - box[0]) / shape[0])) < tolerance and \
                abs(1.0 - ((box[3] - box[1]) / shape[1])) < tolerance:
            return True
        return False
