import os
import sys

PROJECT_ROOT = os.getcwd()[:os.getcwd().index('utils')]
sys.path.append(PROJECT_ROOT)

from utils.files.io_utils import read_filenames, load_dict_from_yaml


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
    print('Found %d images in subset: %s' % (len(img_pairs), SUBSET))


if __name__ == '__main__':
    # Give absolute paths
    OUTPUT_DIR = "/home/andya/external_ssd/wormhole_learning/dataset/testing_handlabeling" \
                 "/night_sampled_rss"
    DATASET_DIR = "/home/andya/external_ssd/wormhole_learning/converted_rosbags"
    FILTER_KEYWORDS = ['day']

    # Use dataset dictionary instead of filter keywords
    DATASET = '/home/andya/external_ssd/wormhole_learning/dataset.yaml'
    SUBSET = ['testing', 'night']

    # Overwrite filter_keywords with dataset file
    # Comment this line out if you want to use the normal keyword filter
    FILTER_KEYWORDS = load_dict_from_yaml(DATASET)[SUBSET[0]][SUBSET[1]]

    MAIN_SENSOR = "RGB"
    AUX_SENSOR = "EVENTS"
    main()
