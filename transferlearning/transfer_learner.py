"""
author: aa
"""

import csv
import os
import time

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageStat
from matplotlib import pyplot as plt

from objdetection.detector.detector import Detector
from objdetection.encoder.encoder_tfrecord_googleapi import EncoderTFrecGoogleApi
from transferlearning.filter.learning_filter import LearningFilter
from utils.io.io_utils import read_filenames
from utils.sheets_interface import GoogleSheetsInterface
from utils.static_helper import load_labels
from utils.visualisation.static_helper import visualize_detections


# Use different back-end to circumvent check for Display variable
# Uncomment if you want to use verbose
# import matplotlib
# matplotlib.use('Agg')


class TransferLearner:
    def __init__(self, flags):
        # Set all flags
        self.flags = flags

        # Load all file names
        self.files = read_filenames(flags.dataset_dir, flags.filter_keywords, flags.main_sensor,
                                    flags.aux_sensor, 'png')

        # Load frozen rgb detector to create annotations
        self._detector = Detector(net_id=flags.net_arch,
                                  labels_net_arch=flags.labels_net,
                                  labels_output=flags.labels_out,
                                  retrieval_thresh=flags.retrieval_thresh)

        # Encoder for tfrecords
        self.labels = load_labels(flags.labels_out)
        self._encoder = EncoderTFrecGoogleApi()
        self.output = os.path.join(flags.dataset_dir, flags.tfrecord_name_prefix + ".tfrecord")

        # Initialize filter
        self._learning_filter = LearningFilter(score_threshold=flags.lf_score_thresh,
                                               min_img_perimeter=flags.min_obj_size,
                                               logstats=True, mode=self.flags.lf_mode,
                                               verbose=False)

        # Analyze the dataset
        self._analysis = self._analyze_dataset(flags, self.files)

        # Image Stats
        self._encoded_mean = []
        self._encoded_std = []

    def transfer(self):
        """
        Loop through images, run inference on main sensor image and encode auxiliary sensor image.
        Writes the transferred labels and bboxes to a new tfrecord.
        :return:
        """
        with tf.python_io.TFRecordWriter(self.output) as writer:
            for count, (file_main_sensor, file_aux_sensor) in enumerate(self.files):
                t_start = time.time()
                img_main = np.array(Image.open(file_main_sensor))
                img_aux = np.array(Image.open(file_aux_sensor))

                obj_detected = self._detector.run_inference_on_img(img_main)

                classes_remapped, scores_remapped, boxes_remapped = self._detector.remap_labels(
                        obj_detected.classes, obj_detected.scores, obj_detected.boxes)

                if len(boxes_remapped) > 0:
                    classes_remapped = np.squeeze(classes_remapped, axis=0)
                    boxes_remapped = np.squeeze(boxes_remapped, axis=0)
                    scores_remapped = np.squeeze(scores_remapped, axis=0)

                # Apply learning filter and boxes from ROI removal if specified
                classes_remapped, boxes_remapped = self._learning_filter.apply(
                        img_main, img_aux, classes_remapped, boxes_remapped)
                classes_remapped, boxes_remapped = self._learning_filter.remove_boxes_from_roi(
                        classes_remapped, boxes_remapped, self.flags.remove_roi,
                        self.flags.remove_shape)

                # Apply preprocessing to the auxiliary image to be encoded
                img_aux = self._image_preprocessor(img_aux)

                # Create instance dict
                instance = {"image":    img_main,
                            "boxes":    boxes_remapped,
                            "labels":   classes_remapped,
                            "filename": file_main_sensor
                            }

                # Encode instance and write example to tfrecord
                tf_example = self._encoder.encode(instance)
                writer.write(tf_example.SerializeToString())
                t_loop = (time.time() - t_start) * 1000
                print("\r[ %i / %i ] Encoded %s in %.1f ms" % (
                    count, len(self.files), os.path.basename(file_main_sensor), t_loop), end="")

                # Stats
                self._encoded_mean.append(img_aux.mean())
                self._encoded_std.append(img_aux.std())

                if self.flags.verbose:
                    obj_detected.classes = classes_remapped
                    obj_detected.boxes = boxes_remapped
                    obj_detected.scores = scores_remapped
                    self._visualize_transfer_step(obj_detected, img_main, img_aux, self.labels)

        # Print Mean & Std of all encoded images
        print('\n================ STATS OF ENCODED IMAGES ================')
        print('Mean: %.2f' % np.mean(self._encoded_mean))
        print('Std:  %.2f' % np.mean(self._encoded_std))

    def _image_preprocessor(self, img_aux):
        """
        Preprocesses auxiliary images if specified and normalizes the image range.
        :param img_aux:
        :return:
        """
        if self.flags.normalize:
            if self.flags.per_image_normalization:
                img_aux = (img_aux - img_aux.mean()) / img_aux.std()
            else:
                img_aux = np.divide(np.subtract(img_aux, self._analysis["mean"]),
                                    self._analysis["stddev"])
            if self.flags.scale_back_using_cv2:
                img_aux = cv2.normalize(img_aux, None, alpha=0, beta=255,
                                        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            else:
                img_aux = ((img_aux * self._analysis["stddev_scale"]) +
                           self._analysis["mean_scale"]).clip(0, 255).astype(np.uint8)
        return img_aux

    def save_statistics(self):
        """
        Save collected statistics in the learning filter to pickled file and pdf
        :return:
        """
        self._learning_filter.stats.save(self.flags.output_dir, self.flags.tfrecord_name_prefix)
        if self.flags.generate_plots:
            self._learning_filter.stats.make_plots(save_plots=self.flags.generate_plots,
                                                   output_dir=self.flags.output_dir,
                                                   filename=self.flags.tfrecord_name_prefix,
                                                   show_plots=self.flags.show_plots)
        # Save dataset stats
        if self.flags.normalize and not self.flags.per_image_normalization:
            stats_file = os.path.join(self.flags.dataset_dir,
                                      self.flags.tfrecord_name_prefix + "_stats.csv")
            data = [['data_type', 'r', 'g', 'b'],
                    ['mean'].extend(self._analysis["mean"]),
                    ['stddev'].extend(self._analysis["stddev"])]
            with open(stats_file, 'w') as fs:
                writer = csv.writer(fs)
                writer.writerows(data)

        # Prepare data for upload to google sheets result page
        values = []
        for idx in range(len(self.labels)):
            number = len(
                self._learning_filter.stats.get_tlscores(label_filt=idx + 1, tl_keep_filt=1))
            diff = len(self._learning_filter.stats.get_tlscores(label_filt=idx + 1, tl_keep_filt=0))
            values.append('%d (%d)' % (number, diff))
        values.append(len(self._learning_filter.stats.get_tlscores()))
        sheets = GoogleSheetsInterface()
        sheets.upload_data('datasets', 'B', 'H', self.flags.tfrecord_name_prefix, values)

    @staticmethod
    def _visualize_transfer_step(obj_detected, img_main, img_aux, labels, use_cv2=False):
        """
        Verbose method
        :param obj_detected:
        :param img_main:
        :param img_aux:
        :param labels:
        :param use_cv2:
        :return:
        """
        img_main_labeled = visualize_detections(img_main, obj_detected, labels=labels)
        img_aux_labeled = visualize_detections(img_aux, obj_detected, labels=labels)
        img_stack = np.hstack((img_main_labeled, img_aux_labeled))
        if use_cv2:
            cv2.imshow('Transfer Learning Step', cv2.cvtColor(img_stack, cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)
        else:
            plt.figure("figure", figsize=(16, 8))
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img_stack)
            plt.show()

    @staticmethod
    def _analyze_dataset(flags, files):
        """
        Analyzes the dataset to its image mean and variance
        :param flags:
        :param files:
        :return:
        """
        mean, stddev, mean_scale, stddev_scale = 0, 0, 0, 0;
        if flags.normalize:
            if not flags.per_image_normalization:
                v_mean = []
                v_stddev = []
                for count, (main, aux) in enumerate(files):
                    img_aux = Image.open(aux)
                    stat = ImageStat.Stat(img_aux)
                    v_mean.append(stat.mean)
                    v_stddev.append(stat.stddev)
                    print("\r[ %.1f %% ] Analyzing dataset" % (count / len(files) * 100), end='')
                mean = np.mean(v_mean, axis=0)
                stddev = np.mean(v_stddev, axis=0)
                print("\r[ 100.0 %% ] Analysis finished:")
                print("\tImages mean: ", mean)
                print("\tImages stddev: ", stddev)

            # Values for scaling back to 0 - 255 while preserving variance
            mean_scale = 127.0
            stddev_scale = 127.0 / flags.confidence_interval
        return {'mean':         mean,
                'stddev':       stddev,
                'mean_scale':   mean_scale,
                'stddev_scale': stddev_scale
                }

    @staticmethod
    def _generate_filename(flags, events_encoding=''):
        """
        :return:
        """
        out_folder = os.path.join(flags.dataset_dir, flags.out_dir)
        name = flags.tfrecord_name + "_" + events_encoding + ".tfrecord"
        return os.path.join(out_folder, name)
