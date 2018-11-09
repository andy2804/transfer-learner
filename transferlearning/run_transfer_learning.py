"""
author: aa & az
"""

import os
import sys
import time
from datetime import timedelta

import tensorflow as tf

from transferlearning.tl_cfgs.static_helper import load_config
from transferlearning.transfer_learner import TransferLearner

PROJECT_ROOT = os.getcwd()[:os.getcwd().index('transferlearning')]
sys.path.append(PROJECT_ROOT)

flags = tf.flags
FLAGS = flags.FLAGS
CONFIG = 'zauron_rgb.yaml'

flags = load_config(flags, CONFIG)

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