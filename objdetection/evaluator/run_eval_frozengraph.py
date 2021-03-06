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

from objdetection.evaluator.eval_frozengraph import run_evaluation


flags = tf.flags
FLAGS = flags.FLAGS

# ================ DIRECTORIES
flags.DEFINE_string('dataset_dir', "/path/to/your/dataset",
                    'The directory where the dataset files are stored as absolute path.')
flags.DEFINE_string('output_dir', "/path/to/your/output",
                    'If any output is being generated it will be saved here')

# ================ FILENAME
flags.DEFINE_list('testfiles', ["TESTSET.tfrecord", ],
                  'Names of the tfrecords containing the test data passed as a list')
flags.DEFINE_string('testname', "NAME OF TEST RUN",
                    'Used in the title of the plot: "Performance metrics on $(arg testname)"')

# ================ NETWORK
flags.DEFINE_integer('network_model', 1,
                     'Network model as defined in the obj_detection class')
flags.DEFINE_string('arch_config', 'default',
                    'Which architecture dictionary to load in nets_ckpt')
flags.DEFINE_string('labels_net_arch', 'label_map.json',
                    'Labels on which the network has been trained')
flags.DEFINE_string('labels_output', 'label_map.json',
                    'Labels against which we want to measure the performance, '
                    'they need to be the same mapping that has been used for the ground-truth.')

# ================ CUDA DEVICES
flags.DEFINE_string('cuda_visible_devices', "0",
                    'Passes argument to cuda visible devices, comma separated values')

# ================ DATASET
flags.DEFINE_integer('num_readers', 2,
                     'Number of readers to decode the tf_records')
flags.DEFINE_integer('num_parallel_calls', 5,
                     'Number of parallel calls of the map function dataset')
flags.DEFINE_integer('batch_size', 1,
                     'The number of samples in each batch.')
flags.DEFINE_integer('prefetch_buffer_factor', 5,
                     'Times the batch_size determines the prefetch size')
flags.DEFINE_bool('normalize_images', False,
                  'Whether or not to normalize validation images (like IR)')
flags.DEFINE_bool('scale_back_using_cv2', False,
                  'Whether or not to scale back images using cv2.normalize() method which will '
                  'discard image mean and variance values. Otherwise normalized images are scaled'
                  'back to range 0 to 255 using mean 127 and stddev derived from confidence_int')

# ================ VERBOSE
flags.DEFINE_string('verbose', '',
                  'Whether or not to visualise the predictions against the ground truth')
flags.DEFINE_bool('make_plot', True,
                  'Make plot out of the results')

# ================ OPTIONS
flags.DEFINE_integer('n_thresholds', 101,
                     'Number of cutoff threshold')
flags.DEFINE_integer('min_obj_size', 0,
                     'Minimum object size (circumference) in pixels, set to 0 to turn it off')
flags.DEFINE_bool('eval_difficult', True,
                  'Whether or not to evaluate object with difficult flag == 1')
flags.DEFINE_bool('publish_result', True,
                  "Wether or not to publish to google sheets.")


def main(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.cuda_visible_devices
    t0 = time.time()
    run_evaluation(FLAGS)
    deltatime = timedelta(seconds=time.time() - t0)
    print("\nConversion completed in:\t", deltatime)


if __name__ == '__main__':
    tf.app.run(main)
