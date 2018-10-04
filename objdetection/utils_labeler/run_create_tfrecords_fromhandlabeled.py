"""
author: aa & az

The labels are made with LabelImg available at: https://github.com/tzutalin/labelImg

This script takes images from a folder (image_src) and the corresponding annotation as xml from
another folder (image_scr_labels) to create tfrecords in the googleAPI format for evaluation.
"""
import os
import sys
import time
from datetime import timedelta

import tensorflow as tf

PROJECT_ROOT = os.getcwd()[:os.getcwd().index('objdetection')]
sys.path.append(PROJECT_ROOT)

from objdetection.meta.utils_labeler.tfrecords_fromhandlabels import \
    create_tfrecords_fromhandlabels

"""
 USAGE:
 --------------------------------------------------------------
 Creates a *.tfrecord dataset from hand-labeled data
 For this to work, the dataset needs to prepared in following
 fashion:

 - Put all the files into same folder makes everything simpler
 - Create a classes.txt file containing all classes in correct order
 - Keep folder structure as follows:
 - /shared_experiments/kaist/hand_labeled/
 - /shared_experiments/kaist/hand_labeled/I001538.xml
 - /shared_experiments/kaist/hand_labeled/I001538_RGB.png
 - /shared_experiments/kaist/hand_labeled/I001538_IR.png
 --------------------------------------------------------------
 If you are not using any tags in the file names set data_filter to None
 Define the image_type to be the type of image to be included
 in the *.tfrecord file
 It is important that the *.xml file has no tags in it, since
 it is used to identify which image to include in the *.tfrecord
 --------------------------------------------------------------
 For renaming files use the run_batch_rename_files script

"""

flags = tf.flags
FLAGS = flags.FLAGS
AVAILABLE_IMAGE_TYPES = ("EVENTS_gauss80", "RGB", "IR")

# ================ DIRECTORIES
flags.DEFINE_string('image_src', "/shared_experiments/kaist/hand_labeling/img",
                    'The directory where the images are stored as absolute path.')
flags.DEFINE_string('image_src_labels', "/shared_experiments/kaist/hand_labeling/hand_labeled",
                    'The directory where the image labels are stored as absolute path')
flags.DEFINE_string('output_dir', "/shared_experiments/kaist/hand_labeling/",
                    'Absolute path for the storing of the generated "*.tfrecord" ')
flags.DEFINE_string('output_file', "TEST.tfrecord", 'Output file name of *.tfrecord')

# todo detangle illumination tags from ego-motion and amount of events
flags.DEFINE_list('data_filter', None, 'Filter for the type of data we want to include')

# ================ GPU MASK
flags.DEFINE_string('cuda_visible_devices', "3",
                    'Passes argument to cuda visible devices, comma separeted values')

# ================ NETWORK
flags.DEFINE_string('labels_map', 'kaist_label_map.json',
                    'Labels on which the network has been trained')

# ================ OPTIONS
flags.DEFINE_string('image_type', "RGB",
                    'Image type to be stored in the "*.tfrecord", should be in '
                    'AVAILABLE_IMAGE_TYPES')
flags.DEFINE_bool('difficult_flag', True,
                  'Wether or not to encode the difficult flag from the PASCAL VOC annotations')


def main(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.cuda_visible_devices
    t0 = time.time()
    create_tfrecords_fromhandlabels(FLAGS)
    deltatime = timedelta(seconds=time.time() - t0)
    print("\nConversion completed in:\t", deltatime)


if __name__ == '__main__':
    tf.app.run()