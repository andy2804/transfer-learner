import os

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from objdetection.detector.detector import ObjectDetected
from objdetection.evaluator.evaluator import EvaluatorFrozenGraph
from utils.files.io_utils import export_transfer_step_img
from utils.visualisation.static_helper import visualize_detections


def _filter_difficult(gt_labels_in, gt_boxes_in, difficult_flag_in, eval_difficult=False):
    """
    If eval_difficult is set to False, all object with difficult_flag value of 1 will
    be removed from gt_labels and gt_boxes
    :param gt_labels_in:
    :param gt_boxes_in:
    :param difficult_flag_in:
    :param eval_difficult:
    :return:
    """
    difficult_flag = difficult_flag_in
    if difficult_flag.size == 0 and gt_labels_in.size > 0:
        difficult_flag = np.expand_dims(np.array([0] * gt_labels_in.shape[1]), axis=0)
    if not eval_difficult:
        keep = np.where(difficult_flag[0, :] == 0)[0]
        gt_labels = gt_labels_in[:, keep]
        gt_boxes = gt_boxes_in[:, keep]
        difficult_flag = difficult_flag_in[:, keep]
    else:
        gt_labels = gt_labels_in
        gt_boxes = gt_boxes_in
    return gt_labels, gt_boxes, difficult_flag


def _filter_object_size(gt_labels_in, gt_boxes_in, difficult_flag_in, img_shape, min_obj_size=0):
    """
    If min_obj_size is greater than 0, the perimeter of each bounding box is calculated.
    If the circumference is smaller than the threshold, the object will be removed from GT.
    :param gt_labels_in:
    :param gt_boxes_in:
    :param difficult_flag_in:
    :param img_shape:
    :param min_obj_size:
    :return:
    """
    cfs = []
    for box in gt_boxes_in[0, :]:
        # Calculate circumference for each object
        y_min, x_min, y_max, x_max = box[0], box[1], box[2], box[3]
        cfs.append(int(2 * img_shape[1] * (y_max - y_min) + 2 * img_shape[2] * (x_max - x_min)))

    # Only keep those with cf greater than threshold
    keep = np.where(np.array(cfs) >= min_obj_size)[0]
    return gt_labels_in[:, keep], gt_boxes_in[:, keep], difficult_flag_in[:, keep]


def _normalize_image(img_in, mean_norm=127.0, confidence_interval=3.0, scale_back_using_cv2=False):
    # Apply normalization to the image to be encoded
    img_norm = (img_in - img_in.mean()) / img_in.std()
    if scale_back_using_cv2:
        img_norm = cv2.normalize(img_norm, None, alpha=0, beta=255,
                                 norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    else:
        stddev_norm = mean_norm / confidence_interval
        img_norm = ((img_norm * stddev_norm) + mean_norm).clip(0, 255).astype(np.uint8)
    return np.expand_dims(img_norm, axis=0)


def run_evaluation(flags):
    """
    Runs the evaluation for the specified network architecture in the run file.
    Additional parameters such as min_obj_size and handling of difficult_flags can be passed.
    Parameters are defined in the run file aswell.
    Results are plotted and saved to local destination, and they are uploaded to
    a Google Evaluation Sheet automatically.
    :param flags:
    :return:
    """
    evaluator = EvaluatorFrozenGraph(net_arch=flags.network_model,
                                     arch_config=flags.arch_config,
                                     labels_net_arch=flags.labels_net_arch,
                                     labels_output=flags.labels_output,
                                     output_dir=flags.output_dir,
                                     n_thresholds=flags.n_thresholds)
    network = evaluator._arch_dict[flags.network_model]
    with evaluator.detection_graph.as_default():
        with tf.Session(graph=evaluator.detection_graph) as sess:
            # "Dataset"
            filenames_placeholder = tf.placeholder(tf.string, shape=[None])
            testfiles = [os.path.join(flags.dataset_dir, f) for f in
                         flags.testfiles]
            dataset = tf.data.TFRecordDataset(filenames_placeholder)

            # Parsing input function takes care of reading and formatting the TFrecords
            dataset = dataset.map(evaluator.parser_input_function)
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
                    print('\r[ %i ] Processing batches...' % batch_count, end='', flush=True)
                    batch_in = sess.run(next_batch)
                    img_in = batch_in['frame']
                    gt_labels = batch_in['gt_labels']
                    gt_boxes = batch_in['gt_boxes']
                    difficult_flag = batch_in['difficult_flag']

                    # Run object detection using selected network architecture
                    (classes, scores, boxes) = sess.run(
                            [evaluator.detection_classes,
                             evaluator.detection_scores,
                             evaluator.detection_boxes],
                            feed_dict={evaluator.image_tensor: img_in})

                    # Remove all difficult object if flags.eval_difficult == False
                    gt_labels, gt_boxes, difficult_flag = _filter_difficult(
                            gt_labels, gt_boxes, difficult_flag, flags.eval_difficult)

                    # Apply min object size filter
                    if flags.min_obj_size > 0:
                        # For GT
                        gt_labels, gt_boxes, difficult_flag = _filter_object_size(gt_labels,
                                                                                  gt_boxes,
                                                                                  difficult_flag,
                                                                                  img_in.shape,
                                                                                  flags.min_obj_size)
                        # For Prediction
                        classes, boxes, scores = _filter_object_size(classes, boxes, scores,
                                                                     img_in.shape,
                                                                     flags.min_obj_size)

                    # Apply normalization to the image if required
                    if flags.normalize_images:
                        img_in = _normalize_image(img_in[0, :],
                                                  scale_back_using_cv2=flags.scale_back_using_cv2)

                    # Remap labels to desired output label map
                    classes_out, scores_out, boxes_out = evaluator.remap_labels(
                            classes, scores, boxes)

                    # Show example if verbose == True
                    if flags.verbose:
                        obj_gt = ObjectDetected('ground-truth', gt_boxes[0, :], None,
                                                gt_labels[0, :], 0)
                        obj_pred = ObjectDetected('prediction', boxes_out[0, :], scores_out[0, :],
                                                  classes_out[0, :], 0)
                        _visualize_transfer_step(flags, network, (obj_gt, obj_pred),
                                                 (img_in[0, :], img_in[0, :]),
                                                 evaluator._labels_output_dict, batch_count)
                        # evaluator.show_example(img_in[0, :],
                        #                        classes_out[0, :],
                        #                        scores_out[0, :],
                        #                        boxes_out[0, :],
                        #                        gt_labels[0, :],
                        #                        gt_boxes[0, :],
                        #                        difficult_flag[0, :])

                    # Update statistics from this batch
                    evaluator.update_evaluation_from_single_batch(
                            classes_out[0, :],
                            scores_out[0, :],
                            boxes_out[0, :],
                            gt_labels[0, :],
                            gt_boxes[0, :])

                    batch_count += 1
                except tf.errors.OutOfRangeError:
                    break
            evaluator.compute_stats()
            if flags.make_plot:
                evaluator.plot_performance_metrics(testname=flags.testname, relative_bar_chart=True)
            evaluator.store_and_publish(filename=flags.testname, min_obj_size=flags.min_obj_size)
            evaluator.print_performance()
    return


def _visualize_transfer_step(flags, network, obj_detected, images, labels, count):
    """
    Verbose method
    :param obj_detected:
    :param images:
    :param labels:
    :param mode:
    :return:
    """
    img_main_labeled = visualize_detections(images[0], obj_detected[0], labels=labels)
    img_aux_labeled = visualize_detections(images[1], obj_detected[1], labels=labels)
    img_stack = np.hstack((img_main_labeled, img_aux_labeled))
    if flags.verbose == 'cv2':
        cv2.imshow('Prediction', img_stack)
        cv2.waitKey(1)
    elif flags.verbose == 'plot':
        plt.figure("figure", figsize=(16, 8))
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img_stack)
        plt.show()
    elif flags.verbose == 'export' and count % 4 == 0:
        export_transfer_step_img(img_stack,
                                 os.path.join(flags.output_dir, '%s_%s' % (network, flags.testname)),
                                 count)
