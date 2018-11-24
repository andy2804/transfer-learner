import os
import shutil
import sys
from random import sample

import numpy as np
from PIL import Image

PROJECT_ROOT = os.getcwd()[:os.getcwd().index('utils')]
sys.path.append(PROJECT_ROOT)

from utils.files.io_utils import read_filenames
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
    img_pairs = read_filenames(dataset_dir, filter, main_sensor, aux_sensor)
    img_pairs_empty, img_pairs_full = [], []

    if biased_sampling:
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
        img_pairs_full = sample(img_pairs_full, int(n_samples * (1 - p_empty_images)))
        img_pairs_empty = sample(img_pairs_empty, int(n_samples * p_empty_images))
    else:
        sampling_step = np.floor_divide(len(img_pairs), n_samples)
        img_pairs_full = img_pairs[::sampling_step]
        print("Selected {:d} images, one each {:d} frames".format(
                len(img_pairs_full), sampling_step))

    for i, pair in enumerate(img_pairs_full + img_pairs_empty):
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        for im in pair:
            if main_sensor in im:
                dst_basename = "%05d_%s.png" % (i, main_sensor)
            else:
                dst_basename = "%05d_%s.png" % (i, aux_sensor)
            dst = os.path.join(output_dir, dst_basename)
            shutil.copyfile(im, dst)
            print("\r[ %d / %d ] Copying %s" % (
            i + 1, len(img_pairs_full + img_pairs_empty), dst_basename), end='')
    print("\nDone!")


if __name__ == '__main__':
    # Give absolute paths
    output_dir = "/home/andya/external_ssd/wormhole_learning/dataset_np/thehive_samples/day_sampled"
    dataset_dir = "/home/andya/external_ssd/wormhole_learning/dataset_np/testing"
    main_sensor = "RGB"
    aux_sensor = "EVENTS"
    filter = ['day']
    n_samples = 2000
    biased_sampling = False
    # Percentage of images which can contain no detectable objects if biased_sampling
    p_empty_images = 0
    # Mask devices
    cuda_visible_devices = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    main()
