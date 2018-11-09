import argparse
import json
import os
from pprint import pprint

import tensorflow as tf

from objdetection.rgb2events.nets import network_factory
from objdetection.meta.metrics import metrics_tf


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
        acc_mean = sum(performance["tp"].values()) / (
                sum(performance["tp"].values()) + sum(
                performance["fp"].values()))
        rec_mean = sum(performance["tp"].values()) / sum(
                performance["n_gt"].values())
        performance['acc'] = acc
        performance['rec'] = rec
        performance['acc_mean'] = acc_mean
        performance['rec_mean'] = rec_mean
    return performance


def main(files, conf_thresh):
    # First, load mapping from integer class ID to sign name string
    with open(PATH2LABELS, "r") as f:
        raw_dict = json.load(f)
        # reformatting with key as int
        labels_dict = {int(k): v for k, v in raw_dict.items()}

    # Launch the graph
    with tf.Graph().as_default(), tf.Session() as sess:
        # "Instantiate" neural network, get relevant tensors
        ssd_net = network_factory.get_network(NETWORK_MODEL)(
                conf_thresh_cutoff=conf_thresh)
        # "Dataset"
        filenames_placeholder = tf.placeholder(tf.string, shape=[None])
        dataset = tf.contrib.data.TFRecordDataset(filenames_placeholder)
        # Parsing input function takes care of reading and formatting the
        # TFrecords
        dataset = dataset.map(ssd_net.parser_input_function)
        dataset = dataset.repeat(1)
        dataset = dataset.shuffle(buffer_size=3)
        dataset = dataset.batch(15)
        iterator = dataset.make_initializable_iterator()
        next_batch = iterator.get_next()
        sess.run(iterator.initializer, feed_dict={filenames_placeholder: files,
                                                  ssd_net.is_training:   False
                                                  })
        # Load trained model
        saver = tf.train.Saver()
        print('Restoring previously trained model at %s' % MODEL_SAVE_PATH)
        saver.restore(sess, MODEL_SAVE_PATH)

        performance = None
        while True:
            try:
                batch_in = sess.run(next_batch)
                perf_out = sess.run(
                        ssd_net.last_batch_performance,
                        feed_dict={ssd_net.events_in:       batch_in["events"],
                                   ssd_net.y_db_conf:       batch_in[
                                                                "y_db_conf"],
                                   ssd_net.y_db_conf_score: batch_in[
                                                                "y_db_conf_score"],
                                   ssd_net.y_db_loc:        batch_in[
                                                                "y_db_loc"],
                                   ssd_net.gt_labels:       batch_in[
                                                                'gt_labels'],
                                   ssd_net.gt_boxes:        batch_in[
                                                                'gt_boxes'],
                                   ssd_net.is_training:     False
                                   })
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
                                "/tfHANDTESTgaus30",
                        action="store_true",
                        help='Directory of input videos/images'
                             'Will run inference on all videos/images in that'
                             ' dir')
    parser.add_argument('--modelckpt_dir',
                        default="/home/ale/datasets/zuriscapes/Logs_official"
                                "/Logs/"
                                "03Dec2017_144140tfTRAINgaus40/175_gaus_model.ckpt",
                        action="store_true",
                        help='Path to where the saved model is stashed')
    parser.add_argument('--network_model',
                        default="ssd_davisSAE_master",
                        action="store_true")
    parser.add_argument('--conf_thresh',
                        default=.5,
                        action="store_true")
    parser.add_argument('--path2labels',
                        default="/home/ale/git_cloned/DynamicVisionTracking"
                                "/objdetection/"
                                "SSDneuromorphic/labels/zauron_label_map.json",
                        action="store_true",
                        help='Path to where the labels for the categories are '
                             'stashed')

    # Get and parse command line options
    args = parser.parse_args()
    INPUT_DIR = args.input_dir
    MODEL_SAVE_PATH = args.modelckpt_dir
    PATH2LABELS = args.path2labels
    CONF_THRESH = args.conf_thresh
    NETWORK_MODEL = args.network_model
    input_files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR)]
    mAP_bool = False
    acc4map, rec4map = [], []
    if not mAP_bool:
        main(input_files, CONF_THRESH)
    else:

        import numpy as np

        N = 6
        # mAP_confvalue = np.linspace(0, 1, N, endpoint=False).tolist()
        mAP_confvalue = []
        mAP_confvalue.extend(np.linspace(.97, .70, N, endpoint=False).tolist())
        mAP_confvalue.extend(np.linspace(.70, .2, N, endpoint=False).tolist())
        for conf_lev in mAP_confvalue:
            a, b = main(input_files, conf_lev)
            acc4map.append(a)
            rec4map.append(b)
        pprint(acc4map)
        pprint(rec4map)
# ax.stackplot(x, y1, y2, y3, labels=labels)
