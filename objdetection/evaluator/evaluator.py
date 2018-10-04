"""
author: aa & az
"""

import os
import pickle
from pprint import pprint

import numpy as np
import pandas as pd
import tensorflow as tf
# import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from objdetection.meta.datasets.encoder_tfrecord_googleapi import EncoderTFrecGoogleApi
from objdetection.meta.detector.detector import ARCH_DICT, Detector
from objdetection.meta.metrics import metrics_np
from objdetection.meta.visualisation.plot_mAP_evaluation import plot_performance_metrics
from objdetection.meta.visualisation.static_helper import \
    visualize_boxes_and_labels_on_image_array
from objdetection.meta.visualisation.static_helper import visualize_stereo_images
from objdetection.rgb2ir.sheets_interface import GoogleSheetsInterface


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
        self._corestats = None
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

    def update_evaluation_from_single_batch(
            self, pr_labels, pr_scores, pr_bboxes, gt_labels, gt_boxes):
        """
        :param pr_labels: predicted
        :param pr_scores: predicted
        :param pr_bboxes: predicted
        :param gt_labels: ground-truth
        :param gt_boxes: ground-truth
        :return:
        """
        # todo add support for batch sizes bigger than ones

        # compute stats
        for thresh in self._corestats:
            n_gt, tp, fp = metrics_np.gather_stats_on_single_batch(
                    pr_labels, pr_scores, pr_bboxes, gt_labels, gt_boxes,
                    self._num_classes, thresh)
            self._update_corestats(ret_thresh=thresh, n_gt=n_gt, tp=tp, fp=fp)
        return

    def compute_stats(self):
        self._corestats = self.compute_acc_rec(self._corestats, self._num_classes)
        self._AP, self._mAP = self.compute_ap(self._corestats, self._thresholds)

    @staticmethod
    def compute_acc_rec(corestats, num_classes):
        """
        Evaluate statistics of detected objects and calculate performance metrics
        according to M. Everingham et. al (https://doi.org/10.1007/s11263-014-0733-5
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

    def _update_corestats(self, ret_thresh, n_gt, tp, fp):
        """
        Update ground truth, true positive and false positive parameters for each class
        :param ret_thresh:
        :param n_gt:
        :param tp:
        :param fp:
        :return:
        """
        for cls in range(1, self._num_classes + 1):
            self._corestats[ret_thresh]['n_gt'][cls] += n_gt[cls]
            self._corestats[ret_thresh]['tp'][cls] += tp[cls]
            self._corestats[ret_thresh]['fp'][cls] += fp[cls]
        return

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
        self._corestats = {t: {s: {c: 0 for c in cls_keys} for s in stats_keys} for t in
                           thresh_keys}
        return

    def plot_performance_metrics(self, testname, relative_bar_chart=True):
        """
        Plots performance metrics using corestats and AP values
        :param testname:
        :return:
        """
        self._plot = plot_performance_metrics(
                [self._corestats], [self._AP],
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
            pickle.dump(self._corestats, fp)

        # also upload a copy to google sheets
        sheet_interface = GoogleSheetsInterface()
        sheet_interface.upload_evaluation_stats(
                self._network_name, filename, self._AP, self._mAP, min_obj_size)
        return

    def print_performance(self):
        print('\nCorestats:')
        pprint(self._corestats)
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