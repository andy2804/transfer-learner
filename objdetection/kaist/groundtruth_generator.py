import os
import time

import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt

from objdetection.kaist.utils_readio import read_filenames
from objdetection.meta.datasets.encoder_tfrecord_googleapi import EncoderTFrecGoogleApi
from objdetection.meta.detector.objdet_frozengraph import DetectionGraph
from objdetection.meta.utils_labeler.static_helper import load_labels
from objdetection.meta.visualisation.static_helper import visualize_rgb_detections


def run(flags):
    files = read_filenames(flags.dataset_dir)
    detector = DetectionGraph(
            arch=flags.network_model,
            labels_net_arch=flags.labels_net,
            labels_output=flags.labels_out)

    labels = load_labels(flags.labels_out)
    encoder = EncoderTFrecGoogleApi()

    out_rgb = os.path.join(flags.dataset_dir,
                           flags.tfrecord_name_prefix + "_rgb" + ".tfrecord")
    out_ir = os.path.join(flags.dataset_dir,
                          flags.tfrecord_name_prefix + "_ir" + ".tfrecord")

    with tf.python_io.TFRecordWriter(out_rgb) as writer_rgb, tf.python_io.TFRecordWriter(
            out_ir) as writer_ir:
        # loop through images, run inference on rgb, encode ir and rgb
        for count, file in enumerate(files):
            t_start = time.time()
            # load rgb for inference
            im_rgb = np.array(Image.open(file[0]))
            im_ir = np.array(Image.open(file[1]))
            obj_detected = detector.run_inference_on_img(im_rgb)

            classes_remapped, scores_remapped, boxes_remapped = detector.remap_labels_2(
                    obj_detected.classes, obj_detected.scores, obj_detected.boxes)

            if len(boxes_remapped) > 0:
                classes_remapped = np.squeeze(classes_remapped, axis=0)
                boxes_remapped = np.squeeze(boxes_remapped, axis=0)
                scores_remapped = np.squeeze(scores_remapped, axis=0)

            instance_rgb = {"image":  im_rgb,
                            "boxes":  boxes_remapped,
                            "labels": classes_remapped
                            }
            instance_ir = {"image":  im_ir,
                           "boxes":  boxes_remapped,
                           "labels": classes_remapped
                           }

            # todo log stats of object detected
            tf_example_rgb = encoder.encode(instance_rgb)
            writer_rgb.write(tf_example_rgb.SerializeToString())
            tf_example_ir = encoder.encode(instance_ir)
            writer_ir.write(tf_example_ir.SerializeToString())
            t_loop = (time.time() - t_start) * 1000
            print("[ %i / %i ] Encoded %s in %.1f ms" % (
                count, len(files), os.path.basename(file[0]), t_loop))

            if flags.verbose:
                obj_detected.classes = classes_remapped
                obj_detected.boxes = boxes_remapped
                obj_detected.scores = scores_remapped

                im_rgb = visualize_rgb_detections(im_rgb, obj_detected, labels=labels)
                plt.imshow(im_rgb)
                plt.show()
