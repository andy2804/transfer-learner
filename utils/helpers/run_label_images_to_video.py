"""
author: aa
"""

import os
import sys

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = os.getcwd()[:os.getcwd().index('utils')]
sys.path.append(PROJECT_ROOT)

from utils.files.io_utils import load_arch_dict, read_filenames, load_dict_from_yaml
from utils.visualisation.static_helper import \
    (visualize_boxes_and_labels_on_image_array,
     add_text_overlay)
from utils.static_helper import load_labels
from objdetection.detector.detector import Detector

NET_ARCH = 90
ARCH_DICT = load_arch_dict('zurich_networks')
RET_THRESH = 0.1
LABELS_NET = "zauron_label_map.json"
LABELS_OUT = "zauron_label_map.json"
INPUT_DIR = "/home/andya/external_ssd/wormhole_learning/converted_rosbags"
OUTPUT_DIR = "/home/andya/external_ssd/wormhole_learning/results/daynight_testing_night_010"
DATASET = "/home/andya/external_ssd/wormhole_learning/dataset.yaml"
SENSOR_NAME = 'RGB'
FILTER_KEYWORDS = load_dict_from_yaml(DATASET)['testing']['night']

# If TESTNAME is None, the script will automatically infer the information
TESTNAME = ARCH_DICT[NET_ARCH]
NORMALIZE = False

CUDA_MASK = "2"
VERBOSE = False


def main():
    """
    Read images, run inference on them and exports them
    :return:
    """
    detector = Detector(net_id=NET_ARCH, arch_config='zurich_networks', labels_net_arch=LABELS_NET,
                        labels_output=LABELS_OUT,
                        retrieval_thresh=RET_THRESH)
    filenames = read_filenames(INPUT_DIR, FILTER_KEYWORDS, SENSOR_NAME, None)
    label_map = load_labels(LABELS_OUT)
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for count, img_path in enumerate(filenames):
        print('\r[ %i ] Processing %s' % (count, os.path.basename(img_path)), end='', flush=True)
        img = np.array(Image.open(img_path))

        # Run Object Detection
        obj_detected = detector.run_inference_on_img(img)
        classes, scores, boxes = detector.remap_labels(
                obj_detected.classes, obj_detected.scores, obj_detected.boxes)

        if len(classes) > 0:
            classes = np.squeeze(classes, axis=0)
            scores = np.squeeze(scores, axis=0)
            boxes = np.squeeze(boxes, axis=0)

        # Add Visualizations
        visualize_boxes_and_labels_on_image_array(img, boxes, classes, scores, label_map,
                                                  use_normalized_coordinates=True, alpha=scores,
                                                  min_score_thresh=RET_THRESH)
        if TESTNAME is None:
            step = int(abs(NET_ARCH) / 10)
            net = ARCH_DICT[NET_ARCH].upper()
            sensor = SENSOR_NAME
            time = 'DAY' if 'day' in img_path else 'NIGHT' if 'night' in img_path else 'N/A'
            overlay_string = 'step:%s    network:%s    sensor:%s    time:%s' % (
                step, net, sensor, time)
        else:
            overlay_string = TESTNAME
        img = add_text_overlay(img, overlay_string, overlay=False, fontsize=14)

        if VERBOSE:
            cv2.imshow('Actual Frame', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)

        # Save image as png
        output_file = os.path.join(OUTPUT_DIR, '%05d.png' % count)
        img = Image.fromarray(img, 'RGB')
        img.save(output_file)

if __name__ == '__main__':
    # main
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_MASK
    main()
