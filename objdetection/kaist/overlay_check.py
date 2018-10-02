import os
import time
from glob import glob

import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt

from objdetection.meta.datasets.encoder_tfrecord_googleapi import EncoderTFrecGoogleApi
from objdetection.meta.detector.objdet_frozengraph import DetectionGraph
from objdetection.meta.utils_labeler.static_helper import load_labels
from objdetection.meta.visualisation.static_helper import visualize_rgb_detections


def run(flags):
    files = _read_filenames(flags.dataset_dir)
    detector = DetectionGraph(arch=flags.net_arch, labels_output=flags.labels_out)
    labels = load_labels(flags.labels_out)
    encoder = EncoderTFrecGoogleApi()

    output = os.path.join(flags.dataset_dir, flags.tfrecord_name_prefix + ".tfrecord")
    with tf.python_io.TFRecordWriter(output) as writer:
        # Loop through images and run inference # todo parallelize?
        for count, file in enumerate(files):
            t_start = time.time()
            img = np.array(Image.open(file))
            obj_detected = detector.run_inference_on_img(img)

            classes_remapped, scores_remapped, boxes_remapped = detector.remap_labels_2(
                    obj_detected.classes, obj_detected.scores, obj_detected.boxes)

            if len(boxes_remapped) > 0:
                classes_remapped = np.squeeze(classes_remapped, axis=0)
                boxes_remapped = np.squeeze(boxes_remapped, axis=0)
                scores_remapped = np.squeeze(scores_remapped, axis=0)

            instance = {"image":  img,
                        "boxes":  boxes_remapped,
                        "labels": classes_remapped
                        }

            # todo log stats of object detected
            tf_example = encoder.encode(instance)
            writer.write(tf_example.SerializeToString())
            t_loop = (time.time() - t_start) * 1000
            print("[ %i / %i ] Encoded %s in %.1f ms" % (
                count, len(files), os.path.basename(file), t_loop))

            if flags.verbose:
                obj_detected.classes = classes_remapped
                obj_detected.boxes = boxes_remapped
                obj_detected.scores = scores_remapped

                img = visualize_rgb_detections(img, obj_detected, labels=labels)
                plt.imshow(img)
                plt.show()


def _read_filenames(dir, filter=None):
    if os.path.exists(dir):
        if filter is not None:
            return [file for path_tuple in os.walk(dir) for file in
                    glob(os.path.join(path_tuple[0], '*.png')) if filter in file]
        else:
            return [file for path_tuple in os.walk(dir) for file in
                    glob(os.path.join(path_tuple[0], '*.png'))]
    else:
        raise IOError("\tThe requested directory does not exists")


def _generate_tfrecord_filename():
    # todo
    pass
