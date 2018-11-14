"""
author: aa & az
"""

import os
import pickle
from pprint import pprint

import numpy as np
import pandas as pd
import tensorflow as tf
from contracts import contract
from matplotlib import pyplot as plt
from scipy import stats

from objdetection.detector.detector import ARCH_DICT, Detector
from objdetection.encoder.encoder_tfrecord_googleapi import EncoderTFrecGoogleApi
from objdetection.metrics import metrics_np
from utils.sheets_interface import GoogleSheetsInterface
from utils.visualisation.plot_mAP_evaluation import plot_performance_metrics
from utils.visualisation.static_helper import \
    visualize_boxes_and_labels_on_image_array
from utils.visualisation.static_helper import visualize_stereo_images


class EvaluatorFrozenGraph(Detector):
    def __init__(self,
                 net_arch,
                 output_dir,
                 labels_net_arch,
                 labels_output=None,
                 plot_title="Performance metrics test",
                 n_thresholds=20):
        """
        #todo
        :param net_arch:
        :param labels_net_arch:
        :param output_dir:
        :param labels_output:
        :param n_thresholds:
        """
        super(EvaluatorFrozenGraph, self).__init__(arch=net_arch,
                                                   labels_net_arch=labels_net_arch,
                                                   labels_output=labels_output)
        self._plot = None
        self.plot_title = plot_title
        self.output_dir = output_dir
        self.decoder = EncoderTFrecGoogleApi()
        self._num_classes = len(self._labels_output_dict)
        self._network_name = ARCH_DICT.get(net_arch)
        # stats
        self._stats = None
        self._mAP = 0
        self._AP = None
        self._init_stats(n_thresholds)

    def parser_input_function(self, example_proto):
        """
        Input function to decode TFrecords and matching of dboxes.
        Data augmentation and matching with default boxes is performed.
        :return: dictionary of elements to feed the network.
        """
        decoded_ex = self.decoder.decode(example_proto)
        gt_boxes = tf.stack(
                [decoded_ex["ymin"], decoded_ex["xmin"], decoded_ex["ymax"], decoded_ex["xmax"]],
                axis=1)
        # todo add padding to support batch extraction
        return {"frame":          decoded_ex["frame"],
                "gt_labels":      decoded_ex["gt_labels"],
                "gt_boxes":       gt_boxes,
                "difficult_flag": decoded_ex["difficult_flag"]
                }

    @contract
    def update_evaluation_from_single_batch(
            self, pr_labels, pr_scores, pr_bboxes, gt_labels, gt_boxes):
        """
        :param pr_labels: predicted
        :type pr_labels: array[N],N>=0
        :param pr_scores: predicted
        :type pr_scores: array[N],N>=0
        :param pr_bboxes: predicted
        :type pr_bboxes: array[Nx4](float),N>=0
        :param gt_labels: ground-truth
        :type gt_labels: *
        :param gt_boxes: ground-truth
        :type gt_boxes: *
        :return:
        """
        # todo add support for batch sizes bigger than ones

        # compute stats
        for thresh in self._stats:
            n_gt, tp, fp = metrics_np.gather_stats_on_single_batch(
                    pr_labels, pr_scores, pr_bboxes, gt_labels, gt_boxes,
                    self._num_classes, thresh)
            self._update_stats(ret_thresh=thresh, n_gt=n_gt, tp=tp, fp=fp)
        return

    def compute_stats(self):
        self._stats = self.compute_acc_rec(self._stats, self._num_classes)
        self._AP, self._mAP = self.compute_ap(self._stats, self._thresholds)

    @staticmethod
    def compute_acc_rec(corestats, num_classes, confidence=None):
        """
        Evaluate statistics of detected objects and calculate performance metrics
        according to M. Everingham et. al (https://doi.org/10.1007/s11263-014-0733-5)
        :return:
        """
        for thresh in corestats:
            if thresh == 1.0:
                for cls in range(1, num_classes + 1):
                    corestats[thresh]['acc'][cls] = 1.0
                    corestats[thresh]['rec'][cls] = 0.0
            else:
                for cls in range(1, num_classes + 1):
                    gt = corestats[thresh]['n_gt'][cls]
                    tp = corestats[thresh]['tp'][cls]
                    fp = corestats[thresh]['fp'][cls]
                    accuracy = EvaluatorFrozenGraph.safe_div(tp, tp + fp)
                    recall = EvaluatorFrozenGraph.safe_div(tp, gt)
                    corestats[thresh]['acc'][cls] = accuracy
                    corestats[thresh]['rec'][cls] = recall
        return corestats

    @staticmethod
    def compute_ap(corestats, thresholds, equal_class_weight=False):
        """
        Calculates the per class average precision
        :param equal_class_weight:
        :return:
        """
        # todo check this function
        num_classes = len(corestats[0]['acc'])
        AP = {c: 0 for c in list(range(1, num_classes + 1))}
        for cls in AP:
            last_rec = 0.0
            ap = 0.0
            for thresh in thresholds[::-1]:
                acc = corestats[thresh]['acc'][cls]
                rec = corestats[thresh]['rec'][cls]
                ap += acc * (rec - last_rec)
                last_rec = rec
            AP[cls] = ap
        # todo introduce equal class weight or not
        mAP = np.mean(list(AP.values()))
        return AP, mAP

    def _update_stats(self, ret_thresh, n_gt, tp, fp):
        """
        Update ground truth, true positive and false positive parameters for each class
        :param ret_thresh:
        :param n_gt:
        :param tp:
        :param fp:
        :return:
        """
        for cls in range(1, self._num_classes + 1):
            self._stats[ret_thresh]['n_gt'][cls] += n_gt[cls]
            self._stats[ret_thresh]['tp'][cls] += tp[cls]
            self._stats[ret_thresh]['fp'][cls] += fp[cls]
        return

    @contract(n_thresholds='int,>0')
    def _init_stats(self, n_thresholds):
        """
        Init the dict of dicts of dict to collect all the basic stats for specified amount of
        retrieval thresholds.
        :param n_thresholds: Number of threshold values to be evaluated [int]
        :return corestats: nested dictionaries
        corestats['threshold']['type of stats']['classes'] = 0
        """
        thresh_keys = np.linspace(0.0, 1.0, n_thresholds)
        np.round(thresh_keys, decimals=2, out=thresh_keys)
        self._thresholds = thresh_keys
        stats_keys = ('n_gt', 'tp', 'fp', 'acc', 'rec')
        cls_keys = list(range(1, self._num_classes + 1))
        self._stats = {t: {s: {c: 0 for c in cls_keys} for s in stats_keys} for t in
                       thresh_keys}
        return

    def plot_performance_metrics(self, testname, relative_bar_chart=True):
        """
        Plots performance metrics using corestats and AP values
        :param relative_bar_chart:
        :param testname:
        :return:
        """
        self._plot = plot_performance_metrics(
                [self._stats], [self._AP],
                self._labels_output_dict,
                testname, relative_bar_chart=relative_bar_chart)

    def store_and_publish(self, filename, min_obj_size=0):
        """
        Saves corestats and AP dictionaries to serialized data and
        publishes evaluation results to the Google Sheet
        :param filename:
        :return:
        """
        # Save the plot as pdf
        try:
            if self._plot is not None:
                plot_file = os.path.join(
                        self.output_dir,
                        self._network_name + '_' + filename + '.pdf')
                self._plot.savefig(plot_file)
        except PermissionError as e:
            e.args += "\n Permission error during saving plots!"

        # pickle corestats and ap results
        pickle_file = os.path.join(self.output_dir,
                                   self._network_name + '_' + filename + '.pickle')
        with open(pickle_file, 'wb') as fp:
            pickle.dump(self._stats, fp)

        # also upload a copy to google sheets
        sheet_interface = GoogleSheetsInterface()
        sheet_interface.upload_evaluation_stats(
                self._network_name, filename, self._AP, self._mAP, min_obj_size)
        return

    def print_performance(self):
        print('\nCorestats:')
        pprint(self._stats)
        print('AP per Class:')
        pprint(self._AP)
        print('mAP:\t%.2f' % self._mAP)
        return

    def show_example(self, frame, plabels, pscores, pbboxes, gtlabels, gtbboxes, difficult_flag):
        """
        Takes all information for one example_proto and visualizes them,
        including Ground Truth, detected Objects and if an object is difficult to detect
        :param frame:
        :param plabels:
        :param pscores:
        :param pbboxes:
        :param gtlabels:
        :param gtbboxes:
        :return:
        """
        frame1, frame2 = np.copy(frame), np.copy(frame)
        visualize_boxes_and_labels_on_image_array(
                frame1, gtbboxes, gtlabels, None, labels=self._labels_output_dict,
                use_normalized_coordinates=True, difficult=difficult_flag)
        visualize_boxes_and_labels_on_image_array(
                frame2, pbboxes, plabels, pscores, labels=self._labels_output_dict,
                use_normalized_coordinates=True)
        stacked_im = visualize_stereo_images([frame1, frame2], titles=("gt", "pred"))
        plt.figure("figure", figsize=(12, 6))
        plt.imshow(stacked_im)
        plt.xticks([])
        plt.yticks([])
        plt.show()
        return

    @staticmethod
    @contract
    def wilson_ci(ns, n, ci=0.95):
        """
        Wilson score interval
        :param ns: number of successes
        :type ns: int,>=0
        :param n: sample size
        :type n: int,>=0
        :param ci: confidence interval
        :type ci: float,>0,<1
        :return: symmetric value
        """
        if n == 0:
            return 0
        z = stats.norm.ppf(1 - (1 - ci) / 2)
        mean = (ns + (z ** 2) / 2) / (n + 2)
        interval = z / (n + z ** 2) * np.sqrt((ns * (n - ns)) / n + (z ** 2) / 4)
        return mean, interval

    @property
    def detection_graph(self):
        return self._detection_graph

    @property
    def detection_classes(self):
        return self._detection_classes

    @property
    def detection_scores(self):
        return self._detection_scores

    @property
    def detection_boxes(self):
        return self._detection_boxes

    @property
    def image_tensor(self):
        return self._image_tensor

    @staticmethod
    def wrap_into_dataframe(d):
        return pd.DataFrame(d)

    @staticmethod
    def safe_div(x, y):
        return 0 if y == 0 else x / y
