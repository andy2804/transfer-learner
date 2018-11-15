"""author:az"""
import os
import sys
import time
from datetime import timedelta
import tensorflow as tf
import matplotlib

matplotlib.use('TkAgg')

PROJECT_ROOT = os.getcwd()[:os.getcwd().index('objdetection')]
sys.path.append(PROJECT_ROOT)
from objdetection.rgb2events.tfrecords_builder.builder_tfrecords import run_conversion

flags = tf.flags
FLAGS = flags.FLAGS

# ========== Directories
flags.DEFINE_string('dataset_dir', "/shared_experiments",
                    'The directory where the encoder files are stored as absolute path.')
flags.DEFINE_string('src_dir', "training/rosbags",  # /converted_rosbags",
                    'Relative path from dataset_dir to the source directory of *.tfrecords.')
flags.DEFINE_string('out_dir', "training/train",
                    'Relative path from dataset_dir to the output directory if any output is '
                    'generated')
flags.DEFINE_string('tfrecord_name', "train",
                    'Output will be "$(tfrecord_name) + id.tfrecord"')
# =========== Gpus mask
flags.DEFINE_string('cuda_visible_devices', "0,1",
                    'Passes argument to cuda visible devices, comma separeted values')
# =========== Filenames mask
flags.DEFINE_list('black_list', [],
                  'All the files containing the listed strings are blacklisted from the conversion')
# ========== Dataset api
flags.DEFINE_integer('num_readers', 15,
                     'Number of readers to decode the tf_records')
flags.DEFINE_integer('num_parallel_calls', 5,
                     'Number of parallel calls of the map function dataset')
flags.DEFINE_integer('batch_size', 32,
                     'The number of samples in each batch.')
flags.DEFINE_integer('shuffle_buffer', 1,
                     'Times the batch size, defines the shuffling buffer of the input pipeline')
flags.DEFINE_integer('prefetch_buffer_factor', 5,
                     'Times the batch_size determines the prefetch size')
# ========== Encoding events parameters
flags.DEFINE_string('weight_fn', 'gaus',
                    'Weighting function for event frame decay')
flags.DEFINE_integer('time_window', 100,
                     'Time window over which we accumulate events into the frame [ms]')
flags.DEFINE_string('labels_map_file', 'resources/labels/zauron_label_map.json',
                    'Labels map file, relative from root project')

# =========== Transfer Learning region parameters
flags.DEFINE_integer('tl_events_time_window', 40,
                     'Time window over which we accumulate events to check the transfer learning '
                     'region [ms]')
flags.DEFINE_integer('tl_keepthresh', 13,
                     'Score threshold over which we retain a training example')
flags.DEFINE_integer('tl_min_img_peri', 80,
                     'Minimum perimeter for the transfer learning region')

# =========== Visualisation and Logging utils
flags.DEFINE_bool('debug', True,
                  'If true it visualises the plots and does not write the .tfrecords files')
flags.DEFINE_bool('generate_plots', False,
                  'Generates and saves plots on the home folder')


def main(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.cuda_visible_devices
    t0 = time.time()
    run_conversion(FLAGS)
    deltatime = timedelta(seconds=time.time() - t0)
    print("\nConversion completed in:\t", deltatime)


if __name__ == '__main__':
    tf.app.run()
