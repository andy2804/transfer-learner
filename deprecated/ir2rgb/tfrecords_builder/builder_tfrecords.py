import os
import time

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt

from objdetection.ir2rgb.tfrecords_builder.learning_filter_tfrecord import TFRecordLearningFilter
from dataset.kaist.utils_readio import read_filenames
from objdetection.meta.datasets.encoder_tfrecord_googleapi import EncoderTFrecGoogleApi
from objdetection.meta.detector.detector import Detector
from objdetection.meta.utils_labeler.static_helper import load_labels
from objdetection.meta.visualisation.static_helper import visualize_boxes_and_labels_on_image_array

__all__ = ['run_conversion']


def run_conversion(flags):
    files = read_filenames(flags.dataset_dir)

    # Load frozen rgb detector to create annotations
    detector = Detector(arch=flags.net_arch,
                        labels_net_arch=flags.labels_net,
                        labels_output=flags.labels_out,
                        retrieval_thresh=flags.retrieval_thresh)

    # Encoder for tfrecords
    labels = load_labels(flags.labels_out)
    encoder = EncoderTFrecGoogleApi()
    output = os.path.join(flags.dataset_dir, flags.tfrecord_name_prefix + ".tfrecord")

    # Initialize filter
    learning_filter = TFRecordLearningFilter(
            score_threshold=flags.lf_score_thresh,
            min_img_perimeter=flags.min_obj_size,
            logstats=True,
            verbose=flags.verbose)

    # Analyze the dataset
    if flags.normalize:
        # Values for scaling back to 0 - 255 while preserving variance
        mean_scale = 127.0
        stddev_scale = 127.0 / flags.confidence_interval

    with tf.python_io.TFRecordWriter(output) as writer:
        # Loop through images, run inference on rgb and encode ir
        for count, (rgb, ir) in enumerate(files):
            t_start = time.time()
            img_rgb = np.array(Image.open(rgb))
            img_ir = np.array(Image.open(ir))

            # Apply normalization to the image to run inference on
            if flags.normalize:
                img_ir = (img_ir - img_ir.mean()) / img_ir.std()
                if flags.scale_back_using_cv2:
                    img_ir = cv2.normalize(img_ir, None, alpha=0, beta=255,
                                           norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                else:
                    img_ir = ((img_ir * stddev_scale) + mean_scale).clip(0, 255).astype(np.uint8)

            obj_detected = detector.run_inference_on_img(img_ir)
            classes, scores, boxes = detector.remap_labels_2(
                    obj_detected.classes, obj_detected.scores, obj_detected.boxes)

            if len(boxes) > 0:
                classes = np.squeeze(classes, axis=0)
                boxes = np.squeeze(boxes, axis=0)
                scores = np.squeeze(scores, axis=0)

            # todo comment changes & adapt rgb2ir tfrecord builder
            if flags.learning_filter:
                difficult = learning_filter.apply_to_tfrec(img_rgb, img_ir, classes, boxes)
            else:
                difficult = np.array([0] * len(boxes))

            if len(difficult) > 0:
                keep = np.where(difficult == 0)[0]
                classes_filtered = classes[keep]
                boxes_filtered = boxes[keep]
            else:
                classes_filtered, boxes_filtered = classes, boxes

            instance = {"image":    img_rgb,
                        "boxes":    boxes_filtered,
                        "labels":   classes_filtered,
                        "filename": rgb
                        }

            tf_example = encoder.encode(instance)
            writer.write(tf_example.SerializeToString())
            t_loop = (time.time() - t_start) * 1000
            print("\r[ %i / %i ] Encoded %s in %.1f ms" % (
                count, len(files), os.path.basename(rgb), t_loop), end="")

            if flags.verbose:
                alpha = np.ones(len(difficult)) - difficult * 0.66
                visualize_boxes_and_labels_on_image_array(img_rgb, boxes, classes, scores, labels,
                                                          use_normalized_coordinates=True,
                                                          difficult=difficult, alpha=alpha)
                visualize_boxes_and_labels_on_image_array(img_ir, boxes, classes, scores, labels,
                                                          use_normalized_coordinates=True)
                img_stack = np.hstack((img_ir, img_rgb))
                plt.figure("figure", figsize=(16, 8))
                plt.xticks([])
                plt.yticks([])
                plt.imshow(img_stack)
                plt.show()

    learning_filter.stats.save(flags.output_dir, flags.tfrecord_name_prefix)
    if flags.generate_plots:
        learning_filter.stats.make_plots(save_plots=flags.generate_plots,
                                         output_dir=flags.output_dir,
                                         filename=flags.tfrecord_name_prefix,
                                         show_plots=flags.show_plots)


def _generate_filename(flags, events_encoding=''):
    """
    :return:
    """
    out_folder = os.path.join(flags.dataset_dir, flags.out_dir)
    name = flags.tfrecord_name + "_" + events_encoding + ".tfrecord"
    return os.path.join(out_folder, name)
