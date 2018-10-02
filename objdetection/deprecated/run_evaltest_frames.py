"""
author: az
"""
import argparse
import os
from pprint import pprint

import tensorflow as tf

from objdetection.meta.performances import metrics_tf
from objdetection.meta.evaluator.evaluator import EvaluatorFrozenGraph


def performance_update(performance, perf_out):
    if performance is None:
        return perf_out
    else:
        # update basic features
        for key2update in ['n_gt', 'tp', 'fp']:
            for clas in performance['n_gt'].keys():
                performance[key2update][clas] += perf_out[key2update][clas]
        # update overall performance
        acc, rec = metrics_tf.compute_acc_rec_np(
                performance['n_gt'], performance['tp'], performance['fp'])
        acc_den = sum(performance["tp"].values()) + sum(
                performance["fp"].values())
        acc_mean = sum(
                performance["tp"].values()) / acc_den if acc_den > 0 else 0
        rec_mean = sum(performance["tp"].values()) / sum(
                performance["n_gt"].values())
        performance['acc'] = acc
        performance['rec'] = rec
        performance['acc_mean'] = acc_mean
        performance['rec_mean'] = rec_mean
    return performance


def main(files, conf_level):
    # First, load mapping from integer class ID to sign name string
    # with open(PATH2LABELS, "r") as f:
    # 	raw_dict = json.load(f)
    # 	# reformatting with key as int
    # 	labels_dict = {int(k): v for k, v in raw_dict.items()}
    # Launch the graph
    net = EvaluatorFrozenGraph(net_arch=NETWORK_MODEL, num_classes=6 + 1,
                               conf_thresh_cutoff=conf_level)
    with net.graph.detection_classes.as_default():
        with tf.Session(graph=net.graph.detection_classes) as sess:
            # "Dataset"
            filenames_placeholder = tf.placeholder(tf.string, shape=[None])
            dataset = tf.contrib.data.TFRecordDataset(filenames_placeholder)
            # Parsing input function takes care of reading and formatting the
            #  TFrecords
            dataset = dataset.map(net.parser_input_function)
            dataset = dataset.repeat(1)
            dataset = dataset.shuffle(buffer_size=10)
            dataset = dataset.batch(1)
            # Initialize input readers
            iterator = dataset.make_initializable_iterator()
            next_batch = iterator.get_next()
            sess.run(iterator.initializer,
                     feed_dict={filenames_placeholder: files})
            performance = None
            while True:
                try:
                    batch_in = sess.run(next_batch)
                    (classes_out, scores_out, boxes_out) = sess.run(
                            [net.detection_classes, net.detection_scores,
                             net.detection_boxes],
                            feed_dict={net.image_in: batch_in["frame"]})
                    perf_out = net.update_evaluation_from_single_batch(classes_out, scores_out,
                                                                       boxes_out,
                                                                       batch_in["gt_labels"],
                                                                       batch_in["gt_boxes"])
                    performance = performance_update(performance, perf_out)
                # pprint(performance)
                except tf.errors.OutOfRangeError:
                    break
            pprint(performance)
            return performance["acc_mean"], performance["rec_mean"]


if __name__ == '__main__':
    # Configure command line options
    parser = argparse.ArgumentParser("Restore trained model and run inference")
    parser.add_argument('--input_dir',
                        default="/home/ale/datasets/zuriscapes"
                                "/tfHANDTESTgaus40",
                        action="store_true",
                        help='Directory of input videos/images'
                             'Will run inference on all videos/images in that'
                             ' dir')
    parser.add_argument('--network_ckpt',
                        default=2,
                        action="store_true")
    parser.add_argument('--conf_thresh',
                        default=.125,
                        action="store_true")
    parser.add_argument('--path2labels',
                        default="/home/ale/git_cloned/DynamicVisionTracking"
                                "/objdetection/"
                                "SSDneuromorphic/labels/zauronscapes_label_map.json",
                        action="store_true",
                        help='Path to where the labels for the categories are '
                             'stashed')

    # Get and parse command line options
    args = parser.parse_args()
    INPUT_DIR = args.input_dir
    PATH2LABELS = args.path2labels
    NETWORK_MODEL = args.network_model
    CONF_THRESH = args.conf_thresh
    input_files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR)]

    mAP_bool = False
    acc4map, rec4map = [], []
    if not mAP_bool:
        main(input_files, CONF_THRESH)
    else:

        import numpy as np

        N = 5
        # mAP_confvalue = np.linspace(0, 1, N, endpoint=False).tolist()
        mAP_confvalue = []
        mAP_confvalue.extend(np.linspace(0.9, .6, N, endpoint=False).tolist())
        mAP_confvalue.extend(np.linspace(.6, .1, N, endpoint=False).tolist())
        for conf_lev in mAP_confvalue:
            a, b = main(input_files, conf_lev)
            acc4map.append(a)
            rec4map.append(b)
        pprint(acc4map)
        pprint(rec4map)
