"""
author: aa & az
"""

import os
import sys
import time
from datetime import timedelta

import tensorflow as tf

PROJECT_ROOT = os.getcwd()[:os.getcwd().index('objdetection')]
sys.path.append(PROJECT_ROOT)
from objdetection.rgb2ir.tfrecords_builder.builder_tfrecords import run_conversion

flags = tf.flags
FLAGS = flags.FLAGS

# ================ LABELS
flags.DEFINE_string('labels_net', 'mscoco_label_map.json',
                    'Labels on which the network has been trained')
flags.DEFINE_string('labels_out', 'kaist_label_map.json',
                    'Labels to be used as output: labels_net will be remapped to labels_out')

# ================ DIRECTORIES
flags.DEFINE_string('dataset_dir', "/shared_experiments/kaist/training/day",
                    'The directory where the encoder files are stored as absolute path.')
flags.DEFINE_string('tfrecord_name_prefix', "KAIST_TRAINING_DAY_rgb_050",
                    'Output will be "$(tfrecord_name).tfrecord"')
flags.DEFINE_string('output_dir', "/shared_experiments/kaist/results/encoder",
                    'If any additional(on top of *.tfrecods) output is being generated it will be '
                    'saved here')

# ================ GPUS MASK

flags.DEFINE_string('cuda_visible_devices', "1",
                    'Passes argument to cuda visible devices, comma separated values')
# ================ SEMI-SUPERVISED LABEL GENERATOR
flags.DEFINE_integer('net_arch', 6,
                     'Passes argument to cuda visible devices, comma separated values')
flags.DEFINE_float('retrieval_thresh', 0.5,
                   'todo')

# ================ PREPROCESSING
flags.DEFINE_bool('normalize', False,
                  'Whether or not to normalize the images in the dataset to zero mean and unit '
                  'variance')
flags.DEFINE_bool('per_image_normalization', True,
                  'Whether or not to normalize per single image or with stats from the whole '
                  'dataset')
flags.DEFINE_float('confidence_interval', 3.0,
                   'Determines the confidence interval for the image values, e.g. 3.0 leads to '
                   '99.7% of the values being kept for the scaling back procedure')
flags.DEFINE_bool('scale_back_using_cv2', False,
                  'Whether or not to scale back images using cv2.normalize() method which will '
                  'discard image mean and variance values. Otherwise normalized images are scaled'
                  'back to range 0 to 255 using mean 127 and stddev derived from confidence_int')
flags.DEFINE_bool('learning_filter', True,
                  'Whether or not to apply the learning filter.')
flags.DEFINE_integer('min_obj_size', 0,
                     'Min Object perimeter needed to transfer learning labels. Put zero to turn '
                     'it off')
flags.DEFINE_integer('lf_score_thresh', 0,
                     'Min Learning Filter Observability Score. Objects below will be discarded'
                     'or flagged as difficult')

# ================ VISUALIZATION AND PLOTS
flags.DEFINE_bool('verbose', False,
                  'Activates verbose mode with visualisation')
flags.DEFINE_bool('generate_plots', True,
                  'Generates and saves plots on the home folder')
flags.DEFINE_bool('show_plots', False,
                  'Wether or not to show generated plots. Put false if running through SSH.')


def main(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.cuda_visible_devices
    t0 = time.time()
    run_conversion(FLAGS)
    deltatime = timedelta(seconds=time.time() - t0)
    print("\nConversion completed in:\t", deltatime)


if __name__ == '__main__':
    tf.app.run()
