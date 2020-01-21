"""
author: aa
"""
import ast
import csv
import os
import sys

import numpy as np

PROJECT_ROOT = os.getcwd()[:os.getcwd().index('WormholeLearning')]
sys.path.append(PROJECT_ROOT)

"""
Takes the exported data from thehive.ai from a .csv file and converts each frame
to a single .xml file with the filename of the original image that has been uploaded to thehive.ai
"""


def _parse_status(status, shape):
    labels = []
    bboxes = []
    for object in status:
        labels.append(object['label'])
        x_min, y_min = np.array((object['p1']['x'], object['p1']['y'])) * shape
        x_max, y_max = np.array((object['p2']['x'], object['p2']['y'])) * shape
        bboxes.append(np.round(np.array([y_min, x_min, y_max, x_max])).astype(int))
    return labels, bboxes


def _create_xml_from_csv(img_name, shape, labels, bboxes):
    img_base = os.path.splitext(img_name)[0]
    with open(os.path.join(OUTPUT_DIR, img_base + '.xml'), 'w') as f:
        line = "<annotation>" + '\n'
        f.write(line)
        line = '\t\t<folder>' + "folder" + '</folder>' + '\n'
        f.write(line)
        line = '\t\t<filename>' + img_name + '</filename>' + '\n'
        f.write(line)
        line = '\t\t<source>\n\t\t<database>' + DATASET + '</database>\n\t</source>\n'
        f.write(line)
        width, height = shape
        line = '\t<size>\n\t\t<width>' + str(width) + '</width>\n\t\t<height>' + str(
                height) + '</height>\n\t'
        line += '\t<depth>3</depth>\n\t</size>'
        f.write(line)
        line = '\n\t<segmented>0</segmented>'
        f.write(line)

        for idx, label in enumerate(labels):
            y_min, x_min, y_max, x_max = bboxes[idx]
            line = '\n\t<object>'
            line += '\n\t\t<name>' + label + '</name>\n\t\t<pose>Unspecified</pose>'
            line += '\n\t\t<truncated>Unspecified</truncated>\n\t\t<difficult>0</difficult>'
            line += '\n\t\t<bndbox>\n\t\t\t<xmin>' + str(x_min) + '</xmin>'
            line += '\n\t\t\t<ymin>' + str(y_min) + '</ymin>'
            line += '\n\t\t\t<xmax>' + str(x_max) + '</xmax>'
            line += '\n\t\t\t<ymax>' + str(y_max) + '</ymax>'
            line += '\n\t\t</bndbox>'
            line += '\n\t</object>\n'
            f.write(line)

        line = "</annotation>" + '\n'
        f.write(line)


def convert():
    """
    Reads an .csv line by line and parses all bounding boxes.
    Creates a PASCAL VOC .xml file with absolute pixel values for the bounding boxes
    :return:
    """
    # Open the file using a csv reader
    with open(INPUT_FILE) as fs:
        reader = csv.reader(fs, delimiter=',')
        print("Opening file:\n%s\n" % INPUT_FILE)

        total_count = sum(1 for line in reader) - 1
        fs.seek(0)
        print("Found %d labels\n" % total_count)

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        print("Parsing labels and saving to:\n%s\n" % OUTPUT_DIR)
        key_dict = {}

        # Iterate through the rows and export a .xml for each frame
        for idx, row in enumerate(reader):
            if len(key_dict) == 0:
                for id, item in enumerate(row):
                    key_dict[item] = id
            else:
                file = row[key_dict['original_filename']]
                status = ast.literal_eval(row[key_dict['status']])
                shape = (int(row[key_dict['media_width']]), int(row[key_dict['media_height']]))

                # Parse Status object
                labels, bboxes = _parse_status(status, shape)
                _create_xml_from_csv(file, shape, labels, bboxes)
                print("\r[ %d / %d ] Exported %s.xml" % (
                idx, total_count, os.path.splitext(file)[0]), end='', flush=True)

    print("\nDone!")


if __name__ == '__main__':
    # Give absolute paths
    INPUT_FILE = "/home/andya/external_ssd/wormhole_learning/dataset/thehive_samples/thehive_day_annotated.csv"
    OUTPUT_DIR = "/home/andya/external_ssd/wormhole_learning/dataset/thehive_samples/thehive_day_labels"
    DATASET = "zurich"

    # Run Conversion
    convert()
