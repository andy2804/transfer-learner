"""
Generate
"""
import os
import sys
import time
from datetime import timedelta

import tensorflow as tf

PROJECT_ROOT = os.getcwd()[:os.getcwd().index('objdetection')]
sys.path.append(PROJECT_ROOT)

from objdetection.kaist.overlay_check import run

flags = tf.flags
FLAGS = flags.FLAGS

# ========== Labels
flags.DEFINE_string('labels_net', 'mscoco_label_map.json',
                    'Labels on which the network has been trained')
flags.DEFINE_string('labels_out', 'zauronscapes_label_map.json',
                    'Labels to be used as output: lapels_net will be remapped to labels_out')
# ========== Directories
# flags.DEFINE_string('dataset_dir', "/shared_experiments",
#                    'The directory where the datasets files are stored as absolute path.')
flags.DEFINE_string('dataset_dir', "/shared_experiments/kaist/training/day",
                    'The directory where the datasets files are stored as absolute path.')
flags.DEFINE_string('tfrecord_name_prefix', "DAY_OVERLAY_CHECK",  # todo modify this
                    'Output will be "$(tfrecord_name).tfrecord"')
# =========== Gpus mask
flags.DEFINE_string('cuda_visible_devices', "4",
                    'Passes argument to cuda visible devices, comma separeted values')
# =========== Ground-truth generator
# Using FasterRCNN Network arch_dict 5
flags.DEFINE_integer('net_arch', -1,
                     'Passes argument to cuda visible devices, comma separeted values')
# =========== Visualisation and verbose
flags.DEFINE_bool('verbose', True,
                  'Activates verbose mode with visualisation')


# ============================================================
# CAN BE DELETED
# ============================================================

def main(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.cuda_visible_devices
    t0 = time.time()
    # run(FLAGS)
    deltatime = timedelta(seconds=time.time() - t0)
    print("\nConversion completed in:\t", deltatime)


if __name__ == '__main__':
    tf.app.run()
