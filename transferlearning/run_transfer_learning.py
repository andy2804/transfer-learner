"""
author: aa & az
"""

import os
import sys
import time
from datetime import timedelta

import tensorflow as tf


PROJECT_ROOT = os.getcwd()[:os.getcwd().index('transferlearning')]
sys.path.append(PROJECT_ROOT)

from utils.files.io_utils import load_dict_from_yaml
from transferlearning.config.static_helper import load_config
from transferlearning.transfer_learner import TransferLearner

flags = tf.flags
FLAGS = flags.FLAGS

# Change this line to load your desired transfer learning configuration
CONFIG = 'default.yaml'

# Uncomment these two lines to use a dataset.yaml instead of the filter keyword
# set in the run CONFIG file.
# DATASET = 'dataset.yaml'
# SUBSET = ['training', 'day']

flags = load_config(flags, CONFIG)

# Overwrite filter_keywords with dataset file
# FLAGS.filter_keywords = load_dict_from_yaml(DATASET)[SUBSET[0]][SUBSET[1]]

# Define which architecture dictionary to use for the detector
flags.DEFINE_string('arch_config', 'zurich_rss_networks',
                    'Which architecture dictionary to load in nets_ckpt')

def main(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.cuda_visible_devices
    t0 = time.time()
    transfer_learner = TransferLearner(FLAGS)
    transfer_learner.transfer()
    transfer_learner.save_statistics()
    deltatime = timedelta(seconds=time.time() - t0)
    print("\nConversion completed in:\t", deltatime)

if __name__ == '__main__':
    tf.app.run()
