import csv
import os
import time

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageStat
from matplotlib import pyplot as plt

from objdetection.kaist.utils_readio import read_filenames
from objdetection.meta.datasets.encoder_tfrecord_googleapi import EncoderTFrecGoogleApi
from objdetection.meta.detector.objdet_frozengraph import DetectionGraph
from objdetection.meta.utils_labeler.static_helper import load_labels
from objdetection.meta.visual.static_helper import visualize_rgb_detections
from objdetection.rgb2ir.tfrecords_builder.learning_filter_tfrecord import TFRecordLearningFilter

__all__ = ['run_conversion']


def run_conversion(flags):
    files = read_filenames(flags.dataset_dir)

    # Load frozen rgb detector to create annotations
    detector = DetectionGraph(arch=flags.net_arch,
                              labels_net_arch=flags.labels_net,
                              labels_output=flags.labels_out,
                              retrieval_thresh=flags.retrieval_thresh)

    # Encoder for tfrecords
    labels = load_labels(flags.labels_out)
    encoder = EncoderTFrecGoogleApi()
    output = os.path.join(flags.dataset_dir, flags.tfrecord_name_prefix + ".tfrecord")

    # Initialize filter
    learning_filter = TFRecordLearningFilter(
            min_img_perimeter=flags.min_obj_size,
            logstats=True,
            verbose=False)

    # Analyze the dataset
    if flags.normalize:
        if not flags.per_image_normalization:
            mean, stddev = _mean_n_var_from_dataset(files)
            # mean = np.array([43.6, 41.7, 43.6])
            # stddev = np.array([23.7, 23.5, 23.7])

        # Values for scaling back to 0 - 255 while preserving variance
        mean_scale = 127.0
        stddev_scale = 127.0 / flags.confidence_interval

    # Image Stats
    encoded_mean = []
    encoded_std = []

    with tf.python_io.TFRecordWriter(output) as writer:
        # Loop through images, run inference on rgb and encode ir
        for count, (rgb, ir) in enumerate(files):
            t_start = time.time()
            img_rgb = np.array(Image.open(rgb))
            img_ir = np.array(Image.open(ir))

            obj_detected = detector.run_inference_on_img(img_rgb)
            # obj_detected = detector.run_inference_on_img(img_ir)

            classes_remapped, scores_remapped, boxes_remapped = detector.remap_labels_2(
                    obj_detected.classes, obj_detected.scores, obj_detected.boxes)

            if len(boxes_remapped) > 0:
                classes_remapped = np.squeeze(classes_remapped, axis=0)
                boxes_remapped = np.squeeze(boxes_remapped, axis=0)
                scores_remapped = np.squeeze(scores_remapped, axis=0)

            if flags.learning_filter:
                classes_remapped, boxes_remapped = learning_filter.apply_to_tfrec(
                        img_rgb, img_ir, classes_remapped, boxes_remapped)

            # Apply normalization to the image to be encoded
            if flags.normalize:
                if flags.per_image_normalization:
                    img_ir = (img_ir - img_ir.mean()) / img_ir.std()
                else:
                    img_ir = np.divide(np.subtract(img_ir, mean), stddev)
                if flags.scale_back_using_cv2:
                    img_ir = cv2.normalize(img_ir, None, alpha=0, beta=255,
                                           norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                else:
                    img_ir = ((img_ir * stddev_scale) + mean_scale).clip(0, 255).astype(np.uint8)

            encoded_mean.append(img_ir.mean())
            encoded_std.append(img_ir.std())

            # Create instance dict
            instance = {"image":    img_rgb,
                        "boxes":    boxes_remapped,
                        "labels":   classes_remapped,
                        "filename": rgb
                        }

            # todo log stats of object detected
            tf_example = encoder.encode(instance)
            writer.write(tf_example.SerializeToString())
            t_loop = (time.time() - t_start) * 1000
            print("\r[ %i / %i ] Encoded %s in %.1f ms" % (
                count, len(files), os.path.basename(rgb), t_loop), end="")

            if flags.verbose:
                obj_detected.classes = classes_remapped
                obj_detected.boxes = boxes_remapped
                obj_detected.scores = scores_remapped

                img_rgb_labeled = visualize_rgb_detections(img_rgb, obj_detected, labels=labels)
                img_ir_labeled = visualize_rgb_detections(img_ir, obj_detected, labels=labels)
                img_stack = np.hstack((img_rgb_labeled, img_ir_labeled))
                plt.figure("figure", figsize=(16, 8))
                plt.xticks([])
                plt.yticks([])
                plt.imshow(img_stack)
                plt.show()

    # Print Mean & Std of all encoded images
    print('\n================ STATS OF ENCODED IMAGES ================')
    print('Mean: %.2f' % np.mean(encoded_mean))
    print('Std:  %.2f' % np.mean(encoded_std))

    learning_filter.stats.save(flags.output_dir, flags.tfrecord_name_prefix)
    if flags.generate_plots:
        learning_filter.stats.make_plots(save_plots=flags.generate_plots,
                                         output_dir=flags.output_dir,
                                         filename=flags.tfrecord_name_prefix,
                                         show_plots=flags.show_plots)

    # Save dataset stats
    if flags.normalize and not flags.per_image_normalization:
        stats_file = os.path.join(flags.dataset_dir, flags.tfrecord_name_prefix + "_stats.csv")
        data = [['data_type', 'r', 'g', 'b'],
                ['mean'].extend(mean),
                ['stddev'].extend(stddev)]
        with open(stats_file, 'w') as fs:
            writer = csv.writer(fs)
            writer.writerows(data)


def _mean_n_var_from_dataset(files):
    """
    Stub for calculating channel mean and standart deviation
    :param files:
    :return: mean value and standart deviation# todo which format?
    """
    v_mean = []
    v_stddev = []
    for count, (rgb, ir) in enumerate(files):
        img_ir = Image.open(ir)
        stat = ImageStat.Stat(img_ir)
        v_mean.append(stat.mean)
        v_stddev.append(stat.stddev)
        print("\r[ %.1f %% ] Analyzing dataset" % (count / len(files) * 100), end='')
    mean = np.mean(v_mean, axis=0)
    stddev = np.mean(v_stddev, axis=0)
    print("\r[ 100.0 %% ] Analysis finished:")
    print("\tImages mean: ", mean)
    print("\tImages stddev: ", stddev)
    return mean, stddev


def _generate_filename(flags, events_encoding=''):
    """
    :return:
    """
    out_folder = os.path.join(flags.dataset_dir, flags.out_dir)
    name = flags.tfrecord_name + "_" + events_encoding + ".tfrecord"
    return os.path.join(out_folder, name)
