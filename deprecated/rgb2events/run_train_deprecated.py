"""
author: az

standard call on rudolf server:
CUDA_VISIBLE_DEVICES=0,1,2 python3 train.py --batch_size 32 --num_epochs 100
--learning_rate 0.0005
---> Shortcuts for root directories:
"/home/ale/Datasets/"
"/media/sdb/DATASETS/"
"""
import os
import sys
import time
from random import shuffle, sample

import tensorflow as tf

sys.path.append(os.getcwd()[:os.getcwd().index('objdetection')])
from objdetection.meta.utils_generic import magic_constants
from objdetection.rgb2events.nets import network_factory

flags = tf.flags
FLAGS = flags.FLAGS
# ========== Directories
flags.DEFINE_string('dataset_dir',
                    "/home/ale/catkin_ws/src/zauron/zauron_eye/recordings/zauron_0.2/",
                    'The directory where the encoder files are stored.')
flags.DEFINE_string('train_dir', "converted_rosbags",
                    'The directory where the encoder files are stored.')
flags.DEFINE_string('dir_to_ckpt',
                    "/home/ale/catkin_ws/src/zauron/zauron_eye/recordings/zauron_0.2/",
                    'The directory where the checkpoints of the model are '
                    'stored.')
# =========== Restore checkpoint
flags.DEFINE_bool('resume_from_ckpt', False,
                  'Whether to resume training or start from scratch')
flags.DEFINE_string('restore_ckpt',
                    "/media/sdb/DATASETS/zuriscapes_official/Logs/"
                    "02Dec2017_223100tfTRAINgaus40/100_gaus_model.ckpt",
                    'The directory where the checkpoints of the model are '
                    'stored.')
# ========== Network architecture
flags.DEFINE_string('network_name', "ssd_net",
                    'The name of the architecture we want to train.')
# ========== Training parameters
flags.DEFINE_integer('batch_size', 1,
                     'The number of samples in each batch.')
flags.DEFINE_float('learning_rate', 0.0001,
                   'The initial learning rate.')
flags.DEFINE_integer('num_epochs', 201,
                     'The number of epochs.')
flags.DEFINE_boolean('is_training', True,
                     'Boolean for training the model, '
                     'it enables variables update')
# ========== Dataset api
flags.DEFINE_integer('num_readers', 2,
                     'Number of readers to decode the tf_records')
flags.DEFINE_integer('shuffle_buffer', 2,
                     'Times the batch size, defines the shuffling buffer of the input pipeline')
flags.DEFINE_float('train_val_ratio', 0.9,
                   'Proportion of split between training and validation, '
                   'suggested: 9/10')
# fixme check what's an appropriate number for prefeatch
flags.DEFINE_integer('prefetch_buffer_size', 10,
                     'todo')


def _get_filenames():
    # Reading filenames and creating split between training and validation data
    train_dir = os.path.join(FLAGS.dataset_dir, FLAGS.train_dir)
    filenames = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    shuffle(filenames)
    train_ex = round(len(filenames) * FLAGS.train_val_ratio)
    files_train = sample(filenames, train_ex)
    files_val = [f for f in filenames if f not in files_train]
    FLAGS.dir_to_ckpt = os.path.join(
            FLAGS.dir_to_ckpt, time.strftime("%Y%b%d_%H%M%S") + FLAGS.train_dir)
    return files_train, files_val


def run_training():
    FLAGS.dir_to_ckpt = os.path.join(
            FLAGS.dir_to_ckpt, time.strftime("%Y%b%d_%H%M%S") + FLAGS.train_dir)
    files_train, files_val = _get_filenames()
    with tf.Graph().as_default(), tf.Session() as sess:
        # Deployment config & options
        objdet_par = magic_constants.DEFAULT_OBJDET_PAR
        net_par = magic_constants.DEFAULT_SSDnet_PARAMS
        learn_par = magic_constants.DEFAULT_LEARN_PARAMS
        events_transform_par = magic_constants.DEFAULT_EVENT_TRANSFORM_PARAMS
        # Global step
        global_step = tf.train.create_global_step()

        # Instantiate graph model object
        network = network_factory.get_network(FLAGS.network_name)
        network = network(objdet_par=objdet_par, net_par=net_par, learn_par=learn_par,
                          events_transform_par=events_transform_par)

        # Dataset Api
        filenames_placeholder = tf.placeholder(tf.string, shape=[None])
        dataset = tf.data.TFRecordDataset(
                filenames_placeholder, num_parallel_reads=FLAGS.num_readers)
        dataset = dataset.map(network.parse_input, num_parallel_calls=FLAGS.num_readers)
        dataset = dataset.repeat(1)
        dataset = dataset.shuffle(buffer_size=FLAGS.batch_size * FLAGS.shuffle_buffer)
        dataset = dataset.batch(FLAGS.batch_size)
        dataset = dataset.prefetch(buffer_size=FLAGS.prefetch_buffer_size)
        iterator = dataset.make_initializable_iterator()
        # network.next_batch() is where the rest of the net is actually instantiated
        network.next_batch(iterator.get_next())
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # Add ops to save and restore all the variables & tensorboard summaries.
        saver = tf.train.Saver()
        # if FLAGS.resume_from_ckpt:
        #     saver.restore(sess, FLAGS.restore_ckpt)
        # else:
        #     sess.run(tf.global_variables_initializer())
        train_summaries = tf.summary.merge_all(key='training')
        valid_summaries = tf.summary.merge_all(key='validation')
        writer_train = tf.summary.FileWriter(
                os.path.join(FLAGS.dir_to_ckpt, "tensorboard_train"),
                graph=sess.graph)
        writer_valid = tf.summary.FileWriter(os.path.join(FLAGS.dir_to_ckpt, "tensorboard_valid"))

        # ====================== Main training/validation loop =================== #
        # ========== Training
        for epoch in range(FLAGS.num_epochs):
            batch_count = 0
            sess.run(iterator.initializer,
                     feed_dict={filenames_placeholder: files_train})
            train_loss = []
            train_request = [network.train_opt, network.loss, train_summaries]
            while True:
                try:
                    # Add summaries only for the first batch
                    summary_bool = epoch % 1 == 0 and batch_count == 0
                    if summary_bool:
                        sess_outputs = sess.run(train_request,
                                                feed_dict={network.is_training: True})
                        writer_train.add_summary(sess_outputs[2], tf.train.get_global_step().eval())
                        writer_train.flush()
                    else:
                        sess_outputs = sess.run(train_request[:2],
                                                feed_dict={network.is_training: True})
                    train_loss.append(sess_outputs[1])
                    batch_count += 1
                except tf.errors.OutOfRangeError:
                    break
            avg_train_loss = sum(train_loss) / len(train_loss)
            max_train_loss, min_train_loss = max(train_loss), min(train_loss)
            # ========= Validation
            v = 0
            sess.run(iterator.initializer,
                     feed_dict={filenames_placeholder: files_val})
            val_loss = []
            outputs_request = [network.loss, valid_summaries]
            while True:
                try:
                    summary_bool = epoch % 1 == 0 and v == 0
                    if summary_bool:
                        sess_outputs = sess.run(outputs_request,
                                                feed_dict={network.is_training: False})
                        writer_valid.add_summary(sess_outputs[1], tf.train.get_global_step().eval())
                        writer_valid.flush()
                    else:
                        sess_outputs = sess.run(outputs_request[:1],
                                                feed_dict={network.is_training: False})
                    val_loss.append(sess_outputs[0])
                    v += 1
                except tf.errors.OutOfRangeError:
                    break
            mean_val_loss = sum(val_loss) / len(val_loss)
            max_val_loss, min_val_loss = max(val_loss), min(val_loss)
            print('==> Epoch: {:d}\t|\tTrain mean: {:.2f} max: {:.2f} min: {:.2f}\t|\t'
                  'Val mean: {:.2f} max: {:.2f} min: {:.2f}'.format(
                    epoch, avg_train_loss, max_train_loss, min_train_loss,
                    mean_val_loss, max_val_loss, min_val_loss))
            # Save the variables to disk at these epochs.
            if epoch in [10, 20, 40, 75, 100, 125, 150, 175, 200]:
                name = str(epoch) + "_gaus_model.ckpt"
                save_path = saver.save(sess,
                                       os.path.join(FLAGS.dir_to_ckpt, name))
                print("Model saved in file: %s" % save_path)


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
