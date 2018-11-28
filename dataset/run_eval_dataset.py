"""
author: az & aa
"""
import os
import sys
import time
from datetime import timedelta

import tensorflow as tf

from dataset.eval_dataset import run_evaluation

PROJECT_ROOT = os.getcwd()[:os.getcwd().index('dataset')]
sys.path.append(PROJECT_ROOT)


flags = tf.flags
FLAGS = flags.FLAGS

# ================ DIRECTORIES
flags.DEFINE_string('input_dir', "/home/andya/external_ssd/wormhole_learning/dataset",
                    'The directory where the dataset files are stored as absolute path.')
flags.DEFINE_string('output_dir', "/home/andya/external_ssd/wormhole_learning/dataset/stats",
                    'If any output is being generated it will be saved here')

# ================ FILENAME
flags.DEFINE_string('filename', "ZURICH_TESTING_NIGHT_V2_rgb_handlabeled.tfrecord",
                    'Names of the tfrecords containing the test data passed as a list')
flags.DEFINE_string('google_sheet', "zurich_dataset",
                    'Title of the worksheet in Google Sheets to upload the results to')

# ================ GPUS MASK
flags.DEFINE_string('cuda_visible_devices', "",  # No GPUS needed for analysis
                    'Passes argument to cuda visible devices, comma separated values')

# ================ NETWORK
flags.DEFINE_string('labels_output', 'zauron_label_map.json',
                    'Labels against which we want to measure the performance, '
                    'they need to be the same mapping that has been used for the ground-truth.')

# ================ DATASET
flags.DEFINE_integer('num_readers', 2,
                     'Number of readers to decode the tf_records')
flags.DEFINE_integer('num_parallel_calls', 5,
                     'Number of parallel calls of the map function dataset')
flags.DEFINE_integer('batch_size', 1,
                     'The number of samples in each batch.')
flags.DEFINE_integer('prefetch_buffer_factor', 5,
                     'Times the batch_size determines the prefetch size')

# ================ VERBOSE
flags.DEFINE_bool('verbose', False,
                  'Whether or not to visualise the predictions against the ground truth')
flags.DEFINE_bool('make_plots', True,
                  'Make plot from results')


def main(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.cuda_visible_devices
    t0 = time.time()
    run_evaluation(FLAGS)
    deltatime = timedelta(seconds=time.time() - t0)
    print("\nAnalysis completed in:\t", deltatime)


if __name__ == '__main__':
    tf.app.run(main)
