"""
author: az
"""
import os
import sys
from random import shuffle
import tensorflow as tf

PROJECT_ROOT = os.getcwd()[:os.getcwd().index('objdetection')]
sys.path.append(PROJECT_ROOT)
from objdetection.meta.utils_events.sae import Sae
from objdetection.meta.utils_generic import magic_constants
from objdetection.rgb2events.tfrecords_builder.builder_tfrecords import ConvertTFrecords
from objdetection.meta.utils_labeler.writer import Writer

flags = tf.flags
FLAGS = flags.FLAGS
REL_LABELS_DIR = "resources/labels/"
TEMP_BLACK_LIST = ["ODS", "ODC"]
# ========== Directories
flags.DEFINE_string('dataset_dir', "/shared_experiments",
                    'The directory where the encoder files are stored as absolute path.')
flags.DEFINE_string('src_dir', "training/rosbags/converted_rosbags",
                    'Relative path from dataset_dir to the source directory of *.tfrecords.')
flags.DEFINE_string('out_dir', "training/extracted_frames_night",
                    'Relative path from dataset_dir to the output directory if any output is '
                    'generated')
# ========== Training/Reading parameters
flags.DEFINE_integer('batch_size', 16,
                     'The number of samples in each batch, it determines only the fetch size from '
                     'disk.')
# ========== Dataset api
flags.DEFINE_integer('num_readers', 15,
                     'Number of readers to decode the tf_records')
flags.DEFINE_integer('num_parallel_calls', 5,
                     'Number of parallel calls of the map function dataset')
flags.DEFINE_integer('shuffle_buffer', 10,
                     'Times the batch size, defines the shuffling buffer of the input pipeline')
flags.DEFINE_integer('prefetch_buffer_factor', 5,
                     'Times the batch_size determines the prefetch size')
# ========== Encoding events parameters
flags.DEFINE_string('weight_fn', 'gaus',
                    'Weighting function for event frame decay')
flags.DEFINE_integer('time_window', 80,
                     'Time window over which we accumulate events into the frame [ms]')
flags.DEFINE_string('labels_map_file', 'resources/labels/zauron_label_map.json',
                    'Labels map file, relative from')
# mask gpus for rudolf todo pass these as parameters with FLAGS
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"


def _get_filenames():
    """
    Reading filenames of TFrecords and shuffling
    :return:
    """
    src_dir = os.path.join(FLAGS.dataset_dir, FLAGS.src_dir)
    filenames = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if
                 f.endswith(".tfrecord") and
                 all([blackflag not in f for blackflag in TEMP_BLACK_LIST])]
    shuffle(filenames)
    return filenames


def run_conversion():
    filenames = _get_filenames()
    with tf.Graph().as_default(), tf.Session() as sess:
        # Deployment config & options
        events_transform_par = magic_constants.EventTransformParameters(
                weight_fn=FLAGS.weight_fn, time_window=FLAGS.time_window)

        # Instantiate graph model object
        converter = ConvertTFrecords(events_transform_par)
        # Encoder
        writer = Writer(output_folder=os.path.join(FLAGS.dataset_dir, FLAGS.out_dir))

        # Dataset Api
        filenames_placeholder = tf.placeholder(tf.string, shape=[None])
        dataset = tf.data.TFRecordDataset(
                filenames_placeholder, num_parallel_reads=FLAGS.num_readers)
        dataset = dataset.map(converter.parse_input, num_parallel_calls=FLAGS.num_parallel_calls)
        dataset = dataset.repeat(1)
        dataset = dataset.shuffle(buffer_size=FLAGS.batch_size * FLAGS.shuffle_buffer)
        dataset = dataset.batch(FLAGS.batch_size)
        dataset = dataset.prefetch(buffer_size=FLAGS.batch_size * FLAGS.prefetch_buffer_factor)
        iterator = dataset.make_initializable_iterator()

        # Event Transform
        ev_transform = Sae(weight_fn=events_transform_par.weight_fn,
                           time_window=events_transform_par.time_window)

        # ========= Visualising after decoding
        sess.run(iterator.initializer,
                 feed_dict={filenames_placeholder: filenames})
        next_batch = iterator.get_next()
        batch_count = 0

        while True:
            try:
                input_prot_out = sess.run(next_batch)
                dynamic_batch_size = input_prot_out["frame_ts"].shape[0]
                for i in range(dynamic_batch_size):
                    events_img = ev_transform.np_transform_events(
                            events=input_prot_out["events"][i, :],
                            to_ts=input_prot_out["frame_ts"][i, 0])

                    writer.dump(events_im=events_img,
                                rgb_im=input_prot_out["frame"][i, :],
                                source_id=input_prot_out["source_id"][i, 0].decode(),
                                events_enc=ev_transform.get_encoding_str())

                batch_count += 1
                print("\rProcessed data instances: {:d}".format(
                        batch_count * FLAGS.batch_size), end='', flush=True)
            except tf.errors.OutOfRangeError:
                break
    print("\nConversion completed!")


def main(_):
    run_conversion()


if __name__ == '__main__':
    tf.app.run()
