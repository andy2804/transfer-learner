"""
author: A. Zanardi

Script deployed to create the event-based labels in a semi-supervised way 
exploiting
pre-trained network
"""
import argparse
import json
import os
import pickle
import sys
from random import shuffle

import tensorflow as tf

sys.path.append(os.getcwd()[:os.getcwd().index('objdetection')])
from objdetection.deprecated import (encoder_tfrecord_deprecated, input_formatter_deprecated,
                                     stats_logger_deprecated, load_recording_deprecated)
from objdetection.meta.detector import objdet_frozengraph
from objdetection.deprecated.detection_filter_deprecated import DetectionFilter

# ========= Extras for plotting
# import numpy as np
# from NeuromorphicDeepLearning.datasets import visualization_utils as 
# vis_util
# from matplotlib import pyplot as plt
# from matplotlib import ticker
# ============
EXAMPLES_count = 0
TFrecord_counter = 0
EXAMPLESperTFrecord = 32
QUEUE_n = 12
BUFFER = [[] for _ in range(QUEUE_n)]


def _generateTFrec_name():
    name = os.path.join(FOLD_OUT_PATH,
                        str(TFrecord_counter).zfill(5) + '.tfrec')
    return name


def _load_pickles(path2pickles):
    pickles_list = []
    for single_path in path2pickles:
        with open(single_path, 'rb') as f:
            try:
                pickles_list.append(pickle.load(f))
            except IOError:
                print("Error loading pickle")
    return pickles_list


def _store(obj_of_interest, category_idx):
    """
    :param obj_of_interest: dictionary containing stuff to be stored
    :param category_idx:
    :return:
    """
    global EXAMPLES_count, EXAMPLESperTFrecord, BUFFER, TFrecord_counter
    example = encoder_tfrecord_deprecated.encode_tfrecord(obj_of_interest)
    BUFFER[EXAMPLES_count % QUEUE_n].append(example)
    # If the buffer is full we store the examples
    if len(BUFFER[-1]) == EXAMPLESperTFrecord:
        for que in range(len(BUFFER)):
            # Open TFrecord
            tf_record_name = _generateTFrec_name()
            with tf.python_io.TFRecordWriter(tf_record_name) as writer:
                shuffle(BUFFER[que])
                for ex in BUFFER[que]:
                    writer.write(ex.SerializeToString())
            BUFFER[que].clear()
            TFrecord_counter += 1

    if EXAMPLES_count % 100 == 0:
        print('Already converted %d examples' % EXAMPLES_count)
    # ========================= Comment out if no plotting is needed 
    # ====================== #
    """
    if EXAMPLES_count % 1 == 0:
        obj_of_interest["events"] = 1 + obj_of_interest["events"]
        obj_of_interest["events"] = np.multiply(
                np.stack([obj_of_interest["events"]] * 3, axis=-1), 
                255 / 2).astype(dtype=np.uint8)
        obj_of_interest["image"] = np.stack([obj_of_interest["image"]] * 3, 
        axis=-1).astype(dtype=np.uint8)
        vis_util.visualize_boxes_and_labels_on_image_array(
                obj_of_interest["image"],
                obj_of_interest["boxes"],
                obj_of_interest["classes"],
                obj_of_interest["scores"],
                category_idx,  # temporary passage of the graph
                use_normalized_coordinates=True,
                line_thickness=1)
        fig = plt.figure(figsize=(20, 8))
        fig.subplots_adjust(wspace=0, hspace=0)
        ax = fig.add_subplot(121)
        ax.xaxis.set_major_locator(ticker.NullLocator())
        ax.yaxis.set_major_locator(ticker.NullLocator())
        ax.imshow(obj_of_interest["image"])
        ax = fig.add_subplot(122)
        ax.xaxis.set_major_locator(ticker.NullLocator())
        ax.yaxis.set_major_locator(ticker.NullLocator())
        ax.imshow(obj_of_interest["events"])
        plt.show()
    """
    # ========================================================= #
    EXAMPLES_count += 1


def _graphInference(rec_fold, graph, loader,
                    formatter, detect_filter, labels_dict, stats_log):
    """
    :param rec_fold:
    :param graph:
    :param loader:
    :param formatter:
    :param detect_filter:
    :param labels_dict:
    :param stats_log:
    :return:
    """
    # begin of main loop
    with graph.detection_classes.as_default():
        with tf.Session(graph=graph.detection_classes) as sess:
            # Load events and frames
            events_dict = loader.load_events(fold=rec_fold)
            frames_list = loader.load_frames(fold=rec_fold)
            # remove first frame as could be too soon
            frames_list = frames_list[1:]
            shuffle(frames_list)
            for s_idx in range(len(frames_list)):
                # Run inference on frame
                detected = graph.run_inference_on_img(
                        os.path.join(rec_fold, frames_list[s_idx][0]),
                        sess=sess)
                # Compute ts from which we crop events
                prev_ts = max(0, frames_list[s_idx][1] - formatter.time_span)
                events = formatter.crop_and_format_events(
                        events_dict, frame_ts=frames_list[s_idx][1],
                        previous_ts=prev_ts)
                detected.update({"events": events})
                # Filter out objects not of interest and encode new labels
                detected, success_flag = detect_filter.process_rawinference(
                        detected)
                if success_flag:
                    _store(obj_of_interest=detected, category_idx=labels_dict)
                    stats_log.update_stats(detected["classes"])


def _humanInference(rec_fold, human_labels_fold, loader,
                    formatter, labels_dict, stats_log):
    # load pickles of handlabeled data
    pickles = os.listdir(human_labels_fold)
    path2pickles = [os.path.join(human_labels_fold, p) for p in pickles]
    loaded_pickles = _load_pickles(path2pickles)
    # load recordings with frames and events
    events_dict = loader.load_events(fold=rec_fold)
    frames_list = loader.load_frames(fold=rec_fold)
    _, rec_name = os.path.split(rec_fold)
    for s_idx in range(len(frames_list)):
        detected = {}
        _, frame_name = os.path.split(frames_list[s_idx][0])
        name_to_match = rec_name + "_" + frame_name
        matching_pickle = [p for p in loaded_pickles if
                           p["frame_name"] == name_to_match]
        if len(matching_pickle) == 1:
            prev_ts = max(0, frames_list[s_idx][1] - formatter.time_span)
            events = formatter.crop_and_format_events(
                    events_dict, frame_ts=frames_list[s_idx][1],
                    previous_ts=prev_ts)
            detected.update({"events": events})
            detected.update({"image": matching_pickle[0]["frame"]})
            detected.update({"boxes": matching_pickle[0]["boxes"].tolist()})
            detected.update({"classes": matching_pickle[0]["ids"].tolist()})
            _store(obj_of_interest=detected, category_idx=labels_dict)
            stats_log.update_stats(detected["classes"])


def run_tfrecords_builder(arch_to_load=2):
    """
    :param arch_to_load:
    :return:
    """
    global TFrecord_counter
    recordings = [os.path.join(RECORDINGS_PATH, r) for r in
                  os.listdir(RECORDINGS_PATH)]
    shuffle(recordings)
    # Maybe download and load frozen graph
    graph = objdet_frozengraph.DetectionGraph(arch=arch_to_load)
    loader = load_recording_deprecated.Loader
    formatter = input_formatter_deprecated.SAE(EVENTS_FORMAT,
                                               time_span=EVENTS_FORMAT_TIMESPAN)
    detection_filter = DetectionFilter(LABEL_MAP_ID,
                                       min_numb_of_objs=0,
                                       for_testing=FOR_TESTING)
    path_to_labels = os.path.join(os.getcwd(),
                                  "labels/zauronscapes_label_map.json")
    with open(path_to_labels, "r") as f:
        raw_dict = json.load(f)
        labels_dict = {int(k): v for k, v in raw_dict.items()}
    stats_log = stats_logger_deprecated.StatsLogger(labels_dict)
    # Process recordings
    for rec_n, record in enumerate(recordings):
        if HANDLABELING:
            _humanInference(rec_fold=record,
                            human_labels_fold=HUMAN_LABELS_FOLD, loader=loader,
                            formatter=formatter, labels_dict=labels_dict,
                            stats_log=stats_log)
        else:
            _graphInference(rec_fold=record, graph=graph, loader=loader,
                            formatter=formatter, detect_filter=detection_filter,
                            labels_dict=labels_dict, stats_log=stats_log)
        print("Recording %d of %d ultimated!" % (rec_n + 1, len(recordings)))
    # === At the end the store leftovers in the buffer
    if len(BUFFER[0]) > 0:
        last_tf_record = []
        for que in BUFFER:
            last_tf_record.extend(que)
        # Open TFrecord
        tf_record_name = os.path.join(FOLD_OUT_PATH,
                                      str(TFrecord_counter).zfill(5) + '.tfrec')
        with tf.python_io.TFRecordWriter(tf_record_name) as writer:
            for ex in last_tf_record:
                writer.write(ex.SerializeToString())
    print("Convertion ultimated!")
    print("Statistics of the converted data:")
    stats_log.print()


if __name__ == "__main__":
    # Parsing command line inputs
    parser = argparse.ArgumentParser(
            "Convert to TFrecords the ourCityscaps dataset")
    parser.add_argument('--keep_thresh', action="store_true", default=0.5,
                        help='Minimum confidence value for detected object to '
                             'be considered ground truth')
    parser.add_argument('--dataset_path', action="store_true",
                        default="/home/ale/datasets/zuriscapes"
                                "/hand_labeling_night/",
                        help='full path to the dataset')
    parser.add_argument('--events_formatter', action="store_true",
                        default="gaus",
                        help='how to format events')
    parser.add_argument('--for_testing', action="store_true", default=True,
                        help='If the records are for testing only, no motion '
                             'filter is applied. '
                             'More in general we remove stochastic elements.')
    parser.add_argument('--events_acc_ts', action="store_true", default=40,
                        help='time span over which accumulate events [ms]')
    parser.add_argument('--recordings_folder', action="store_true",
                        default="uzhNIGHT",
                        help='folders that contains all the saccades')
    parser.add_argument('--prefix_outfolder', action="store_true", default="tf",
                        help='prefix name of the folder to be used as output')
    parser.add_argument('--from_handlabeling', action="store_true",
                        default=True,
                        help='prefix name of the folder to be used as output')
    parser.add_argument('--handlabeling_fold', action="store_true",
                        default="/home/ale/datasets/zuriscapes"
                                "/hand_labeling_night/pickle",
                        help='path to pickled hand labeled data')
    args = parser.parse_args()
    # ========
    DATASET_PATH = args.dataset_path
    RECORDINGS_PATH = os.path.join(DATASET_PATH, args.recordings_folder)
    FOLD_OUT_PATH = os.path.join(args.dataset_path,
                                 args.prefix_outfolder +
                                 args.recordings_folder +
                                 args.events_formatter + str(
                                         args.events_acc_ts))
    KEEP_THRESH = args.keep_thresh
    EVENTS_FORMAT = "_11" + args.events_formatter
    EVENTS_FORMAT_TIMESPAN = args.events_acc_ts / 1000
    LABEL_MAP_ID = {0:  0, 1: 3, 2: 2, 3: 5, 4: 6, 5: 4, 6: 4, 7: 4, 8: 1, 9: 0,
                    10: 0
                    }
    FOR_TESTING = args.for_testing
    HANDLABELING = args.from_handlabeling
    HUMAN_LABELS_FOLD = args.handlabeling_fold
    assert os.path.exists(RECORDINGS_PATH)
    if not os.path.exists(FOLD_OUT_PATH):
        os.mkdir(FOLD_OUT_PATH)
    else:
        print("The output folder already exists! Sure you wanna overwrite?")
    run_tfrecords_builder()
    print("Elaboration compleated!")
