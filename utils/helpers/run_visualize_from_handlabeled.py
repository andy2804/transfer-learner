"""
author: aa
"""

import os
import sys

import cv2
from PIL import Image

PROJECT_ROOT = os.getcwd()[:os.getcwd().index('WormholeLearning')]
sys.path.append(PROJECT_ROOT)

import xml.etree.ElementTree as ET
from utils.static_helper import load_labels
from utils.tfrecords_fromhandlabels import _recursive_parse_xml_to_dict, dict_to_tf_instance
from utils.visualisation.static_helper import visualize_boxes_and_labels_on_image_array

"""
The labels are made with LabelImg available at: https://github.com/tzutalin/labelImg

This script takes images from a folder (image_src) and the corresponding annotation as xml from
another folder (image_scr_labels) to create visualizations of the bounding boxes.
"""


def list_files(input_dir, filter):
    if os.path.isdir(input_dir):
        files = [os.path.join(path_tuple[0], file) for path_tuple in os.walk(input_dir) for
                 file in path_tuple[2] if all([s in file for s in filter]) and FILE_TYPE in file]
        return files


def visualize():
    """
    Takes in images and xml labels in PASCAL VOC format. They must have the same filenames.
    Visualizes the bounding boxes on the images and saves them in the output folder.
    :return:
    """
    image_files = list_files(INPUT_DIR, FILTER_KEYWORDS)
    classes = load_labels(LABEL_MAP)
    class_names = {classes[i]['name']: i for i in list(range(1, len(classes) + 1))}
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for idx, img_path in enumerate(image_files):
        # Parse XML Files
        xml_file = img_path.strip(FILE_TYPE) + '.xml'
        xml_tree = ET.parse(xml_file).getroot()
        xml_data = _recursive_parse_xml_to_dict(xml_tree)
        instance = dict_to_tf_instance(xml_data['annotation'], img_path, class_names)
        visualize_boxes_and_labels_on_image_array(instance['image'], instance['boxes'],
                                                  instance['labels'], None, classes,
                                                  use_normalized_coordinates=True, line_thickness=2)

        # Show Output
        cv2.imshow('Visualization', cv2.cvtColor(instance['image'], cv2.COLOR_BGR2RGB))
        cv2.waitKey(5)
        print("\r[ %d / %d ] Showing %s" % (idx + 1, len(image_files), os.path.basename(img_path)),
              end='')

        # Save image as png
        output_file = os.path.join(OUTPUT_DIR, os.path.basename(img_path))
        img = Image.fromarray(instance['image'], 'RGB')
        img.save(output_file)

    print("\nDone!")


if __name__ == '__main__':
    # Give absolute paths
    INPUT_DIR = "/home/andya/external_ssd/wormhole_learning/dataset_np/thehive_samples/night_hive_examples"
    OUTPUT_DIR = "/home/andya/external_ssd/wormhole_learning/dataset_np/thehive_samples/night_hive_examples/visualization"
    FILTER_KEYWORDS = ['OVERLAY']
    FILE_TYPE = '.png'

    # Specify Label Map
    LABEL_MAP = 'zauron_label_map.json'

    # Run Visualization
    visualize()
