"""
author: aa
"""

import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from objdetection.meta.datasets.encoder_tfrecord_googleapi import EncoderTFrecGoogleApi
from objdetection.meta.utils_labeler.static_helper import load_labels
from objdetection.meta.visualisation.static_helper import visualize_boxes_and_labels_on_image_array
from objdetection.rgb2events.tfrecords_builder.stats import TLStatistician
from objdetection.rgb2ir.sheets_interface import GoogleSheetsInterface

DECODER = EncoderTFrecGoogleApi()


def _input_parser(example_proto):
    """
    Input function to decode TFrecords and matching of dboxes.
    Data augmentation and matching with default boxes is performed.
    :return: dictionary of elements to feed the network.
    """
    decoded_ex = DECODER.decode(example_proto)
    gt_boxes = tf.stack(
            [decoded_ex["ymin"], decoded_ex["xmin"], decoded_ex["ymax"], decoded_ex["xmax"]],
            axis=1)

    return {"frame":          decoded_ex["frame"],
            "gt_labels":      decoded_ex["gt_labels"],
            "gt_boxes":       gt_boxes,
            "difficult_flag": decoded_ex["difficult_flag"]
            }


def run_evaluation(flags):
    """
    Evaluates datasets for objects detected and uploads them to Google Sheet
    :param flags:
    :return:
    """
    stats = TLStatistician(tl_score_threshold=0)
    labels = load_labels(flags.labels_output)
    sheets = GoogleSheetsInterface()

    with tf.Session() as sess:
        # "Dataset"
        filenames_placeholder = tf.placeholder(tf.string, shape=[None])
        testfiles = [os.path.join(flags.dataset_dir + flags.src_dir, flags.filename)]
        dataset = tf.data.TFRecordDataset(filenames_placeholder)

        # Parsing input function takes care of reading and formatting the TFrecords
        dataset = dataset.map(_input_parser)
        dataset = dataset.repeat(1)
        dataset = dataset.batch(flags.batch_size)

        # Initialize input readers
        iterator = dataset.make_initializable_iterator()
        next_batch = iterator.get_next()
        sess.run(iterator.initializer, feed_dict={filenames_placeholder: testfiles})
        batch_count = 0
        while True:
            try:
                # Read in next batch
                print('\r[ %i ] Analysing batches...' % batch_count, end='', flush=True)
                batch_in = sess.run(next_batch)
                img_in = batch_in['frame'][0, :]
                gt_labels = batch_in['gt_labels'][0, :]
                gt_boxes = batch_in['gt_boxes'][0, :]
                difficult_flag = batch_in['difficult_flag'][0, :]

                if len(difficult_flag) == 0 and len(gt_labels) > 0:
                    difficult_flag = [0] * len(gt_labels)

                # Add objects to stats
                for idx, class_id in enumerate(gt_labels):
                    stats.append_obj_stats(
                            label=class_id,
                            ymin=gt_boxes[idx][0],
                            xmin=gt_boxes[idx][1],
                            h=gt_boxes[idx][2] - gt_boxes[idx][0],
                            w=gt_boxes[idx][3] - gt_boxes[idx][1],
                            tl_score=0,
                            tl_difficult=abs(difficult_flag[idx] - 1))

                # Show example if verbose == True
                if flags.verbose:
                    img = np.copy(img_in)
                    visualize_boxes_and_labels_on_image_array(
                            img, gt_boxes, gt_labels, None, labels=labels,
                            use_normalized_coordinates=True, difficult=difficult_flag)
                    plt.imshow(img)
                    plt.show()

                batch_count += 1
                stats.n_instances += 1

            except tf.errors.OutOfRangeError:
                break

        # Create plots to be stored in output folder
        output_name = os.path.splitext(flags.filename)[0]
        if flags.make_plots:
            stats.make_plots(save_plots=True, output_dir=flags.output_dir,
                             filename=output_name, show_plots=flags.verbose, labels_dict=labels)

        # Prepare data for upload to google sheets result page
        values = []
        for idx in range(len(labels)):
            number = len(stats.get_tlscores(label_filt=idx + 1, tl_keep_filt=1))
            diff = len(stats.get_tlscores(label_filt=idx + 1, tl_keep_filt=0))
            values.append('%d (%d)' % (number, diff))
        values.append(len(stats.get_tlscores()))
        sheets.upload_data('datasets', 'B', 'H', output_name, values)
    return
