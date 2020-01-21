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


def visualize():
    """
    Takes in images and xml labels in PASCAL VOC format. They must have the same filenames.
    Visualizes the bounding boxes on the images and saves them in the output folder.
    :return:
    """
    image_files = [f for f in os.listdir(IMG_INPUT_DIR) if
                   f.endswith(".png") and all(key in f for key in FILTER_KEYWORDS)]
    image_files.sort()
    xml_files = [f for f in os.listdir(XML_INPUT_DIR) if f.endswith(".xml")]
    xml_files.sort()
    classes = load_labels(LABEL_MAP)
    class_names = {classes[i]['name']: i for i in list(range(1, len(classes) + 1))}
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for idx, (img_name, xml_name) in enumerate(zip(image_files, xml_files)):
        # Parse XML Files
        xml_file = os.path.join(XML_INPUT_DIR, xml_name)
        xml_tree = ET.parse(xml_file).getroot()
        xml_data = _recursive_parse_xml_to_dict(xml_tree)

        # Create tfRecord dict instance
        img_file = os.path.join(IMG_INPUT_DIR, img_name)
        instance = dict_to_tf_instance(xml_data['annotation'], img_file, class_names)
        visualize_boxes_and_labels_on_image_array(instance['image'], instance['boxes'],
                                                  instance['labels'], None, classes,
                                                  use_normalized_coordinates=True, line_thickness=2)

        # Show Output
        if VERBOSE:
            cv2.imshow('Visualization', cv2.cvtColor(instance['image'], cv2.COLOR_BGR2RGB))
            cv2.waitKey(5)
        print("\r[ %d / %d ] Processing %s" % (idx + 1, len(image_files), os.path.basename(img_file)),
              end='')

        # Save image as png
        output_file = os.path.join(OUTPUT_DIR, os.path.basename(img_file))
        img = Image.fromarray(instance['image'], 'RGB')
        img.save(output_file)

    print("\nDone!")


if __name__ == '__main__':
    # Give absolute paths
    IMG_INPUT_DIR = "/home/andya/external_ssd/wormhole_learning/dataset/thehive_samples/day_sampled"
    XML_INPUT_DIR = "/home/andya/external_ssd/wormhole_learning/dataset/thehive_samples" \
                    "/thehive_day_labels"
    OUTPUT_DIR = "/home/andya/external_ssd/wormhole_learning/dataset/thehive_samples/day_sampled" \
                 "/visualization"
    FILTER_KEYWORDS = ['RGB']
    FILE_TYPE = '.png'

    # Specify Label Map
    LABEL_MAP = 'zauron_label_map.json'

    # Verbose
    VERBOSE = False

    # Run Visualization
    visualize()
