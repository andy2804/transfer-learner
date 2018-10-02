"""
author: aa
"""

import os
import sys

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = os.getcwd()[:os.getcwd().index('objdetection')]
sys.path.append(PROJECT_ROOT)

from objdetection.meta.evaluator.eval_frozengraph import _normalize_image
from objdetection.meta.visualisation.static_helper import \
    (visualize_boxes_and_labels_on_image_array,
     add_text_overlay)
from objdetection.meta.visualisation.vis_helper import compose_media_from_frames
from objdetection.meta.utils_labeler.static_helper import load_labels
from objdetection.meta.detector.objdet_frozengraph import DetectionGraph, ARCH_DICT


def _read_filenames_from_subfolders(dir):
    assert os.path.isdir(dir)
    subfolders = os.listdir(dir)
    filenames = []
    for subfolder in subfolders:
        img_dir = os.path.join(dir, subfolder)
        filenames.extend(
                sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if
                        f.endswith(".png")]))
    return filenames


def main():
    """
    Read images, run inference on them and export it to a .mp4 video
    :return:
    """
    detector = DetectionGraph(arch=NET_ARCH, labels_net_arch=LABELS_NET, labels_output=LABELS_OUT,
                              retrieval_thresh=RET_THRESH)
    filenames = _read_filenames_from_subfolders(DATASET_DIR)
    label_map = load_labels(LABELS_OUT)
    frames_out = []

    for count, img_path in enumerate(filenames):
        print('\r[ %i ] Processing %s' % (count, os.path.basename(img_path)), end='', flush=True)
        img = np.array(Image.open(img_path))
        if NORMALIZE:
            img = np.squeeze(_normalize_image(img), axis=0)

        # Run Object Detection
        obj_detected = detector.run_inference_on_img(img)
        classes, scores, boxes = detector.remap_labels_2(
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
            sensor = 'RGB' if 'rgb' in img_path else 'IR' if 'ir' in img_path else 'N/A'
            time = 'DAY' if 'day' in img_path else 'NIGHT' if 'night' in img_path else 'N/A'
            overlay_string = 'step:%s    network:%s    sensor:%s    time:%s' % (
                step, net, sensor, time)
        else:
            overlay_string = TESTNAME
        img = add_text_overlay(img, overlay_string, overlay=False, fontsize=14)
        frames_out.append(img)
        if VERBOSE:
            cv2.imshow('Actual Frame', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)

    retval = (str('%.2f') % RET_THRESH).replace('.', '')
    output_file = os.path.join(OUTPUT_DIR, '%s_%s_RET%s.mp4' % (step, net, retval))
    compose_media_from_frames(frames_out, 20, output_file)


if __name__ == '__main__':
    NET_ARCH = -21
    RET_THRESH = 0.10
    LABELS_NET = "kaist_label_map.json"
    LABELS_OUT = "kaist_label_map.json"
    DATASET_DIR = "/shared_experiments/kaist/demo/img_ir"
    OUTPUT_DIR = "/shared_experiments/kaist/demo"

    # If TESTNAME is None, the script will automatically infer the information
    TESTNAME = None
    NORMALIZE = True

    CUDA_MASK = "0"
    VERBOSE = True

    # main
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_MASK
    main()
