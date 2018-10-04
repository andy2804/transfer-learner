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

from objdetection.kaist.groundtruth_generator import run

flags = tf.flags
FLAGS = flags.FLAGS

# ========== Labels
flags.DEFINE_string('labels_net', 'mscoco_label_map.json',
                    'Labels on which the network has been trained')
flags.DEFINE_string('labels_out', 'kaist_label_map.json',
                    'Labels to be used as output: lapels_net will be remapped to labels_out')
# ========== Directories
flags.DEFINE_string('dataset_dir', "/shared_experiments/kaist/training/night",
                    'The directory where the datasets files are stored as absolute path.')
flags.DEFINE_string('tfrecord_name_prefix', "KAIST_TRAINING_NIGHT_ZAURONSCAPES_test",
                    'Output will be "$(tfrecord_name).tfrecord"')
# =========== Ground-truth generator
# Using NAS Network arch_dict 6
flags.DEFINE_integer('network_model', 6,
                     'Passes argument to cuda visible devices, comma separeted values')
# =========== Gpus mask
flags.DEFINE_string('cuda_visible_devices', "0",
                    'Passes argument to cuda visible devices, comma separeted values')
# =========== Visualisation and verbose
flags.DEFINE_bool('verbose', True,
                  'Activates verbose mode with visualisation')


def main(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.cuda_visible_devices
    t0 = time.time()
    run(FLAGS)
    deltatime = timedelta(seconds=time.time() - t0)
    print("\nConversion completed in:\t", deltatime)


if __name__ == '__main__':
    tf.app.run()
