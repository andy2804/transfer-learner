"""
author: az & aa
"""
import os
import sys
import time
from datetime import timedelta

import tensorflow as tf

PROJECT_ROOT = os.getcwd()[:os.getcwd().index('objdetection')]
sys.path.append(PROJECT_ROOT)

from objdetection.kaist.eval_dataset import run_evaluation

flags = tf.flags
FLAGS = flags.FLAGS

# ================ DIRECTORIES
flags.DEFINE_string('dataset_dir', "/shared_experiments/kaist/",
                    'The directory where the dataset files are stored as absolute path.')
flags.DEFINE_string('src_dir', "training/day_n_night/",
                    'Relative path from dataset_dir to the source directory of *.tfrecords.')
flags.DEFINE_string('output_dir', "/shared_experiments/kaist/results/datasets",
                    'If any output is being generated it will be saved here')

# ================ FILENAME
flags.DEFINE_string('filename', "KAIST_TRAINING_NIGHT_rgb_from_ir035_scorethresh_015.tfrecord",
                    'Names of the tfrecords containing the test data passed as a list')

# ================ GPUS MASK
flags.DEFINE_string('cuda_visible_devices', "",  # No GPUS needed for analysis
                    'Passes argument to cuda visible devices, comma separated values')

# ================ NETWORK
flags.DEFINE_string('labels_output', 'kaist_label_map.json',
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
