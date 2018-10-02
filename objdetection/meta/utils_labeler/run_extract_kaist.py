import os
import shutil
import sys
from random import sample

import numpy as np
from PIL import Image

PROJECT_ROOT = os.getcwd()[:os.getcwd().index('objdetection')]
sys.path.append(PROJECT_ROOT)
from objdetection.kaist.utils_readio import read_filenames
from objdetection.meta.detector.objdet_frozengraph import DetectionGraph


def main():
    """
    Extract pairs of rgb and ir images from the kaist dataset. A pretrained net can be run
    over the images to check if there is supposed to be something. This option is available via
    "only_rich_imgs" and the parameter "p_empty_imgs" which controls the ratio of samples from the
    allegedly empty ones. No checks on the number of samples are performed, make sure you require
    a reasonable amount.
    :return:
    """
    img_pairs = read_filenames(dataset_dir)
    img_pairs_empty, img_pairs_full = [], []

    if biased_sampling:
        detector = DetectionGraph(arch=2)
        for img_pair in img_pairs[:]:
            img = np.array(Image.open(img_pair[0]))
            obj_detected = detector.run_inference_on_img(img)
            if not obj_detected.scores.size:
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
        for im in pair:
            dst_basename = "I{:06d}".format(i)
            if "lwir" in im:
                dst_basename += "_ir"
            dst_basename += ".png"
            dst = os.path.join(output_dir, dst_basename)
            shutil.copyfile(im, dst)


if __name__ == '__main__':
    # Give absolute paths
    output_dir = "/shared_experiments/kaist/tmp"
    dataset_dir = "/shared_experiments/kaist/testing/night"
    n_samples = 2000
    biased_sampling = False
    # Percentage of images which can contain no detectable objects if biased_sampling
    p_empty_images = 0.1
    # Mask devices
    cuda_visible_devices = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    main()
