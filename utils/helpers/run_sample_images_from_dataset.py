import os
import shutil
import sys
from random import sample

import numpy as np
from PIL import Image

PROJECT_ROOT = os.getcwd()[:os.getcwd().index('utils')]
sys.path.append(PROJECT_ROOT)

from utils.files.io_utils import read_filenames, load_dict_from_yaml
from objdetection.detector.detector import Detector


def main():
    """
    Extract pairs of rgb and ir images from the given dataset. A pretrained net can be run
    over the images to check if there is supposed to be something. This option is available via
    "only_rich_imgs" and the parameter "p_empty_imgs" which controls the ratio of samples from the
    allegedly empty ones. No checks on the number of samples are performed, make sure you require
    a reasonable amount.
    :return:
    """
    img_pairs = read_filenames(DATASET_DIR, FILTER_KEYWORDS, MAIN_SENSOR, AUX_SENSOR)
    img_pairs_empty, img_pairs_full = [], []

    if BIASED:
        detector = Detector(labels_output='zauron_label_map.json')
        for img_pair in img_pairs[:]:
            img = np.array(Image.open(img_pair[0]))
            obj_detected = detector.run_inference_on_img(img)
            classes_remapped, scores_remapped, boxes_remapped = detector.remap_labels(
                    obj_detected.classes, obj_detected.scores, obj_detected.boxes)
            if not classes_remapped.size:
                img_pairs_empty.append(img_pair)
            else:
                img_pairs_full.append(img_pair)
            print("\rSo far found {:d} empty images and {:d} with something".format(
                    len(img_pairs_empty), len(img_pairs_full)), end="", flush=True)
        img_pairs_full = sample(img_pairs_full, int(N_SAMPLES * (1 - P_EMPTY)))
        img_pairs_empty = sample(img_pairs_empty, int(N_SAMPLES * P_EMPTY))
    else:
        sampling_step = np.floor_divide(len(img_pairs), N_SAMPLES)
        img_pairs_full = img_pairs[::sampling_step]
        print("Selected {:d} images, one each {:d} frames".format(
                len(img_pairs_full), sampling_step))

    for i, pair in enumerate(img_pairs_full + img_pairs_empty):
        if not os.path.isdir(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        for im in pair:
            if MAIN_SENSOR in im:
                dst_basename = "%05d_%s.png" % (i, MAIN_SENSOR)
            else:
                dst_basename = "%05d_%s.png" % (i, AUX_SENSOR)
            dst = os.path.join(OUTPUT_DIR, dst_basename)
            shutil.copyfile(im, dst)
            print("\r[ %d / %d ] Copying %s" % (
            i + 1, len(img_pairs_full + img_pairs_empty), dst_basename), end='')
    print("\nDone!")


if __name__ == '__main__':
    # Give absolute paths
    OUTPUT_DIR = "/home/andya/external_ssd/wormhole_learning/dataset/thehive_samples/night_sampled"
    DATASET_DIR = "/home/andya/external_ssd/wormhole_learning/converted_rosbags"
    FILTER_KEYWORDS = ['night']

    # Use dataset dictionary instead of filter keywords
    DATASET = '/home/andya/external_ssd/wormhole_learning/dataset.yaml'
    SUBSET = ['testing', 'night']

    # Overwrite filter_keywords with dataset file
    # Comment this line out if you want to use the normal keyword filter
    FILTER_KEYWORDS = load_dict_from_yaml(DATASET)[SUBSET[0]][SUBSET[1]]

    MAIN_SENSOR = "RGB"
    AUX_SENSOR = "EVENTS"
    N_SAMPLES = 2000
    BIASED = False
    # Percentage of images which can contain no detectable objects if biased_sampling
    P_EMPTY = 0
    # Mask devices
    CUDA_VISIBLE_DEVICES = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    main()
