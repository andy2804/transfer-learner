"""
author: aa
"""

import json
import os

ROOT_DIR = "zauron/"
LABELS_DIR = "resources/labels/"
LABELS_PATH = os.path.join(os.getcwd()[:os.getcwd().index(ROOT_DIR)], ROOT_DIR, LABELS_DIR)


def load_labels(json_file):
    """
    Load labels from json file
    :param json_file:
    :return:
    """
    json_file = os.path.join(LABELS_PATH, json_file)
    with open(json_file, "r") as fs:
        raw_dict = json.load(fs)
    return {int(k): v for k, v in raw_dict.items()}
