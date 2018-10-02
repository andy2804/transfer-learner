import argparse
import json
import os
import time
from random import shuffle

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib import ticker
from matplotlib.backends.backend_pdf import PdfPages

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
import sys

sys.path.append(os.getcwd()[:os.getcwd().index('objdetection')])
from utils_visualisation import fun_general as vis_util
from objdetection.rgb2events.nets import ssd_common, network_factory
from pprint import pprint


def generate_output(files, mode):
    """
    Generate annotated images, videos, or sample images, based on mode
    """
    # First, load mapping from integer class ID to sign name string
    with open(PATH2LABELS, "r") as f:
        raw_dict = json.load(f)
        # reformatting with key as int
        labels_dict = {int(k): v for k, v in raw_dict.items()}

    # Launch the graph
    with tf.Graph().as_default(), tf.Session() as sess:
        # "Instantiate" neural network, get relevant tensors
        ssd_net = network_factory.get_network(NETWORK_MODEL)()
        # "Dataset"
        filenames_placeholder = tf.placeholder(tf.string, shape=[None])
        dataset = tf.contrib.data.TFRecordDataset(filenames_placeholder)
        # Parsing input function takes care of reading and formatting the
        # TFrecords
        dataset = dataset.map(ssd_net.parser_input_function)
        dataset = dataset.repeat(1)
        dataset = dataset.shuffle(buffer_size=3)
        dataset = dataset.batch(1)
        iterator = dataset.make_initializable_iterator()
        next_batch = iterator.get_next()
        sess.run(iterator.initializer, feed_dict={filenames_placeholder: files,
                                                  ssd_net.is_training:   False
                                                  })
        # Load trained model
        saver = tf.train.Saver()
        print('Restoring previously trained model at %s' % MODEL_SAVE_PATH)
        saver.restore(sess, MODEL_SAVE_PATH)
        graph = tf.get_default_graph()

        time_list = []
        if mode == 'image':
            while True:
                try:
                    batch_in = sess.run(next_batch)
                    t0 = time.time()
                    events_out, classes_out, boxes_out, scores_out, loss_out, \
                    perf_out = \
                        sess.run([ssd_net.final_events, ssd_net.final_classes,
                                  ssd_net.final_boxes, ssd_net.final_scores,
                                  ssd_net.loss, ssd_net.last_batch_performance
                                  ],
                                 feed_dict={
                                     ssd_net.events_in:       batch_in[
                                                                  "events"],
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
                                     ssd_net.is_training:     True
                                 })
                    i2disp = 0
                    # print("Loss: %f -- forward pass time: %f [s]" % (
                    # loss_out, t1))
                    classes_out = classes_out[i2disp, :]
                    scores_out = scores_out[i2disp, :]
                    boxes_out = boxes_out[i2disp, :]
                    events_out = events_out[i2disp, :, :, :]
                    gt_labels = batch_in["gt_labels"][i2disp]
                    gt_boxes = batch_in["gt_boxes"][i2disp, :]
                    image = np.squeeze(batch_in["frame"][i2disp, :, :])
                    image = np.stack([image] * 3, axis=-1)
                    vis_util.visualize_boxes_and_labels_on_image_array(
                            image,
                            gt_boxes,
                            gt_labels,
                            None,
                            labels_dict,
                            use_normalized_coordinates=True,
                            line_thickness=1)
                    vis_util.visualize_boxes_and_labels_on_image_array(
                            events_out,
                            boxes_out,
                            classes_out,
                            scores_out,
                            labels_dict,
                            min_score_thresh=ssd_net.conf_thresh_cutoff,
                            use_normalized_coordinates=True,
                            line_thickness=1)
                    time_list.append(time.time() - t0)
                    print("mean %f" % (sum(time_list) / len(time_list)))
                    fig = plt.figure()
                    fig.subplots_adjust(wspace=0, hspace=0)
                    ax = fig.add_subplot(121)
                    ax.xaxis.set_major_locator(ticker.NullLocator())
                    ax.yaxis.set_major_locator(ticker.NullLocator())
                    ax.imshow(image)
                    ax = fig.add_subplot(122)
                    ax.xaxis.set_major_locator(ticker.NullLocator())
                    ax.yaxis.set_major_locator(ticker.NullLocator())
                    ax.imshow(events_out)
                    pprint(perf_out)
                    print(loss_out)
                    plt.show()
                    with PdfPages(
                            '/home/ale/Pictures/foo_perf_temp.pdf') as pdf:
                        # As many times as you like, create a figure fig and
                        # save it:
                        pdf.savefig(figure=fig, bbox_inches='tight')

                except tf.errors.OutOfRangeError:
                    break
                finally:
                    plt.close()
        elif mode == 'fast':
            fig = plt.figure()
            im = plt.imshow(np.zeros([180, 240, 3], dtype=np.uint8), vmin=0,
                            vmax=255)
            plt.show(block=False)
            while True:
                try:
                    batch_in = sess.run(next_batch)
                    t0 = time.time()
                    events_out, classes_out, boxes_out, scores_out = \
                        sess.run([ssd_net.fast_events, ssd_net.fast_classes,
                                  ssd_net.fast_boxes, ssd_net.fast_scores,
                                  ], feed_dict={
                            ssd_net.events_in:   batch_in["events"],
                            ssd_net.is_training: False
                        })
                    if not (scores_out is None):
                        vis_util.visualize_boxes_and_labels_on_image_array(
                                events_out,
                                boxes_out,
                                classes_out,
                                scores_out,
                                labels_dict,
                                min_score_thresh=ssd_net.conf_thresh_cutoff,
                                use_normalized_coordinates=True,
                                line_thickness=1)
                    time_list.append(time.time() - t0)
                    print("mean %f" % (sum(time_list) / len(time_list)))
                    im.set_data(events_out)
                    fig.canvas.draw()
                except tf.errors.OutOfRangeError:
                    break
        elif mode == 'checkencoding':
            print("Works ONLY with single batch size!")
            while True:
                loc_all = tf.reshape(ssd_net.y_db_loc,
                                     shape=[-1, ssd_net.dboxes_tot_n, 4])
                xreal, yreal, wreal, hreal = ssd_common.decode_db(loc_all,
                                                                  ssd_net.dboxes_lookup_tensor,
                                                                  ssd_net.objdet_par.prior_scaling)
                xmin, ymin, xmax, ymax = \
                    ssd_common.bboxes_center_format_to_minmax(
                            xreal, yreal, wreal, hreal)
                enc_boxes = tf.stack([ymin, xmin, ymax, xmax], axis=2)
                enc_classes = tf.squeeze(ssd_net.y_db_conf, axis=0)
                enc_boxes = tf.squeeze(enc_boxes, axis=0)
                indices2keep = tf.squeeze(tf.where(tf.greater(enc_classes, 0)),
                                          axis=1)
                enc_classes = tf.gather(enc_classes, indices2keep)
                enc_boxes = tf.gather(enc_boxes, indices2keep)
                try:
                    batch_in = sess.run(next_batch)
                    encoded_classes, encoded_boxes = sess.run(
                            [enc_classes, enc_boxes],
                            feed_dict={ssd_net.y_db_conf: batch_in["y_db_conf"],
                                       ssd_net.y_db_loc:  batch_in["y_db_loc"]
                                       })
                    # pprint(boxes_out)
                    i2disp = 0
                    # print("Loss: %f -- forward pass time: %f [s]" % (
                    # loss_out, t1))
                    gt_labels = batch_in["gt_labels"][i2disp]
                    gt_boxes = batch_in["gt_boxes"][i2disp, :]
                    events_im = np.squeeze(batch_in["events"][i2disp, :, :, :])
                    events_im = events_im * 255 / 2 + 255 / 2
                    events_im = np.stack([events_im] * 3, axis=-1).astype(
                            dtype=np.uint8)
                    image = np.squeeze(batch_in["frame"][i2disp, :, :, :])
                    image = np.stack([image] * 3, axis=-1).astype(
                            dtype=np.uint8)
                    vis_util.visualize_boxes_and_labels_on_image_array(
                            image,
                            gt_boxes,
                            gt_labels,
                            None,
                            labels_dict,
                            use_normalized_coordinates=True,
                            line_thickness=1)
                    vis_util.visualize_boxes_and_labels_on_image_array(
                            events_im,
                            encoded_boxes,
                            encoded_classes,
                            None,
                            labels_dict,
                            use_normalized_coordinates=True,
                            line_thickness=1)
                    fig = plt.figure(figsize=(20, 8))
                    # TODO plot image with gt boxes, maybe add a final_image
                    # in the graph?!
                    ax = fig.add_subplot(121)
                    ax.imshow(image)
                    ax = fig.add_subplot(122)
                    ax.imshow(events_im)
                    plt.show()
                except tf.errors.OutOfRangeError:
                    break
        elif mode == 'display_kernels':
            kern_11 = graph.get_tensor_by_name(
                    'ssd_davis/conv_1_1/kernel:0').eval()
            kern_12 = graph.get_tensor_by_name(
                    'ssd_davis/conv_1_2/kernel:0').eval()
            bias_22 = graph.get_tensor_by_name(
                    'ssd_davis/conv_2_2/bias:0').eval()
            kern_42 = graph.get_tensor_by_name(
                    'ssd_davis/conv_4_2/kernel:0').eval()
            bias_42 = graph.get_tensor_by_name(
                    'ssd_davis/conv_4_2/bias:0').eval()
            kern_72 = graph.get_tensor_by_name(
                    'ssd_davis/conv_7_2/kernel:0').eval()
            bias_72 = graph.get_tensor_by_name(
                    'ssd_davis/conv_7_2/bias:0').eval()
            kern_fm4 = graph.get_tensor_by_name(
                    'block4_featuremap/conv_conf/kernel:0').eval()
            kern_fm5 = graph.get_tensor_by_name(
                    'block5_featuremap/conv_conf/kernel:0').eval()
            kern_fm6 = graph.get_tensor_by_name(
                    'block6_featuremap/conv_conf/kernel:0').eval()
            kern_fm7 = graph.get_tensor_by_name(
                    'block7_featuremap/conv_conf/kernel:0').eval()
            kern_fm8 = graph.get_tensor_by_name(
                    'block8_featuremap/conv_conf/kernel:0').eval()

            fig1, axs = plt.subplots(8, 8, figsize=(12, 12))
            axs = axs.ravel()
            to_disp = kern_11
            n_ker = 32
            in_ch = 0
            vmin = np.min(to_disp)
            vmax = np.max(to_disp)
            for i in range(n_ker):
                im = axs[i].matshow(np.squeeze(to_disp[:, :, in_ch, i]),
                                    cmap='hot',
                                    vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=axs.ravel().tolist())
            fig2, axs = plt.subplots(8, 4, figsize=(12, 8))
            axs = axs.ravel()
            n_ker = 32
            in_ch = 0
            for i in range(n_ker):
                im = axs[i].matshow(np.squeeze(to_disp[:, :, in_ch, i]),
                                    cmap='hot',
                                    vmin=vmin, vmax=vmax)

            plt.colorbar(im, ax=axs.ravel().tolist())
            plt.show(fig1)
            plt.show(fig2)
        # todo everything
        else:
            raise ValueError('Invalid mode: %s' % mode)


if __name__ == '__main__':
    # Configure command line options
    parser = argparse.ArgumentParser("Restore trained model and run inference")
    parser.add_argument('--input_dir',
                        default="/home/ale/datasets/zuriscapes"
                                "/tfsomeTRAINgaus40",
                        action="store_true",
                        help='Directory of input videos/images'
                             'Will run inference on all videos/images in that'
                             ' dir')
    parser.add_argument('--modelckpt_dir',
                        default="/home/ale/datasets/zuriscapes/Logs_official"
                                "/Logs/"
                                "03Dec2017_120153tfTRAINgaus40/75_gaus_model.ckpt",
                        action="store_true",
                        help='Path to where the saved model is stashed')
    parser.add_argument('--path2labels',
                        default="/home/ale/git_cloned/DynamicVisionTracking"
                                "/objdetection/"
                                "SSDneuromorphic/labels"
                                "/zauronscapes_label_map.json",
                        action="store_true",
                        help='Path to where the labels for the categories are stashed')
    parser.add_argument('--network_model',
                        default="ssd_davisSAE_master",
                        action="store_true")
    parser.add_argument('--mode',
                        default='image',
                        action="store_true",
                        help='Operating mode, could be "image", "video"')

    # Get and parse command line options
    args = parser.parse_args()
    INPUT_DIR = args.input_dir
    MODE = args.mode
    MODEL_SAVE_PATH = args.modelckpt_dir
    PATH2LABELS = args.path2labels
    NETWORK_MODEL = args.network_model
    ALLOWED_MODES = ('checkencoding', 'image', 'display_kernels')

    if MODE not in ALLOWED_MODES:
        raise ValueError('Invalid mode: %s' % MODE)

    input_files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR)]
    shuffle(input_files)
    generate_output(input_files, MODE)
