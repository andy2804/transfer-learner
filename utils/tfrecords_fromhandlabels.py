"""
author: aa & az
"""
import os
import xml.etree.ElementTree as ET

import numpy as np
import tensorflow as tf
from PIL import Image

from objdetection.encoder.encoder_tfrecord_googleapi import EncoderTFrecGoogleApi
from utils.static_helper import load_labels


def dict_to_tf_instance(data, img_path, class_names):
    """
    Reads an xml dict and converts it to an instance ready to be encoded
    :param class_names:
    :param img_path:
    :param data: dict of xml parsed by _recursive_parse_xml_to_dict
    :return:
    """
    image = np.array(Image.open(img_path))

    width = int(data['size']['width'])
    height = int(data['size']['height'])

    boxes = []
    labels = []
    difficult = []
    if 'object' in data:
        for obj in data['object']:
            ymin = float(obj['bndbox']['ymin']) / height
            xmin = float(obj['bndbox']['xmin']) / width
            ymax = float(obj['bndbox']['ymax']) / height
            xmax = float(obj['bndbox']['xmax']) / width
            box = np.array([ymin, xmin, ymax, xmax])
            boxes.append(box)
            labels.append(class_names[obj['name']])
            difficult.append(int(obj['difficult']))
        boxes = np.vstack((boxes))
    else:
        boxes = np.ndarray((0, 4))
    labels = np.asarray(labels)
    difficult = np.asarray(difficult)
    instance = {'image':          image,
                'boxes':          boxes,
                'labels':         labels,
                'difficult_flag': difficult
                }

    return instance


def _recursive_parse_xml_to_dict(xml):
    """Recursively parses XML contents to python dict.
    We assume that `object` tags are the only ones that can appear
    multiple times at the same level of a tree.
    Args:
      xml: xml tree obtained by parsing XML file contents using lxml.etree
    Returns:
      Python dictionary holding XML contents.
    """
    if not xml:
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = _recursive_parse_xml_to_dict(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


# todo adapt to correctly check for tags
def _filter_input(input_list, filter_tags):
    """
    Filters out all files from list not satisfying filter tags
    Note iterator for list needs to have [:] as otherwise while removing
    objects the iterator might skip items from list due to list being shorter afterwards
    :param input_list:
    :param filter_tags:
    :return:
    """
    for object in input_list[:]:
        if not any(tag in object for tag in filter_tags):
            input_list.remove(object)
    return input_list


def _load_classes(folder, classes_txt):
    cls_file = os.path.join(folder, classes_txt[0])
    with open(cls_file) as f:
        classes = f.readlines()
    return [x.strip() for x in classes]


def create_tfrecords_fromhandlabels(flags):
    """
    Recursively creates an tfrecord for each image that has been labeled
    Can be filtered by tags set in the flags
    :param flags:
    :return:
    """
    labels_xml = [f for f in os.listdir(flags.image_src_labels) if f.endswith(".xml")]
    print('Found %i labeled objects' % len(labels_xml))
    encoder = EncoderTFrecGoogleApi()

    # Check if object satisfies filter features
    if flags.data_filter is not None:
        labels_xml = _filter_input(labels_xml, flags.data_filter)
    classes = load_labels(flags.labels_map)
    class_names = {classes[i]['name']: i for i in list(range(1, len(classes) + 1))}
    print('Found %i objects satisfying condition: %s' % (len(labels_xml), flags.data_filter))

    if flags.output_file is not None:
        output = os.path.join(flags.output_dir, flags.output_file)
    else:
        file = '_'.join(flags.data_filter)
        extension = '_%s.tfrecord' % flags.image_type
        output = os.path.join(flags.output_dir, file + extension)

    # Parse labels and create tfrecord example
    try:
        with tf.python_io.TFRecordWriter(output) as writer:
            # Loop through objects and create tfrecords
            for count, label_xml in enumerate(labels_xml):
                file = os.path.splitext(label_xml)[0]

                xml_file = os.path.join(flags.image_src_labels, label_xml)
                xml_tree = ET.parse(xml_file).getroot()
                xml_data = _recursive_parse_xml_to_dict(xml_tree)

                extension = '%s.png' % file
                img_path = os.path.join(flags.image_src, extension)
                instance = dict_to_tf_instance(xml_data['annotation'], img_path, class_names)

                example = encoder.encode(instance, flags.difficult_flag)
                writer.write(example.SerializeToString())
                print('\r[ %i / %i ] %s' % (count + 1, len(labels_xml), os.path.basename(file)),
                      end="")
    except IOError as e:
        e.args += ('Failed attempting to serialize tfRecord file: %s' % file)
    return
