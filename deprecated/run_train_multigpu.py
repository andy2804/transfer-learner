"""
author: az
inspired from:
https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10
/cifar10_multi_gpu_train.py
"""
import os
import sys
import time
from random import shuffle, sample

import tensorflow as tf

sys.path.append(os.getcwd()[:os.getcwd().index('objdetection')])
from objdetection.meta.utils_generic import model_deploy, magic_constants
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
                    "/media/sdb/DATASETS/zuriscapes_official/Logs/",
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
flags.DEFINE_integer('num_gpus', 3,
                     'Number of gpus on which we want to create clones')
flags.DEFINE_integer('num_readers', 2,
                     'Number of readers to decode the tf_records')
flags.DEFINE_integer('shuffle_buffer', 2,
                     'Times the batch size, defines the shuffling buffer of the input pipeline')
flags.DEFINE_float('train_val_ratio', 0.9,
                   'Proportion of split between training and validation, '
                   'suggested: 9/10')
# fixme check what's an appropriate number for prefeatch
flags.DEFINE_integer('prefetch_buffer_size', 10,
                     'Number of gpus on which we want to create clones')


def run_training_multigpu():
    files_train, files_val = _get_filenames()
    with tf.Graph().as_default(), tf.device('/cpu:0'), \
         tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        # Deployment config & options
        net_options = magic_constants.DEFAULT_OBJDET_PAR
        learn_params = magic_constants.DEFAULT_LEARN_PARAMS
        events_transform_params = magic_constants.DEFAULT_EVENT_TRANSFORM_PARAMS
        # is_training = FLAGS.is_training
        deploy_config = model_deploy.DeploymentConfig(
                num_clones=FLAGS.num_gpus)  # fixme can add other options

        # Instantiate graph model object
        network = network_factory.get_network(FLAGS.network_name)
        network = network(net_options, learn_params, events_transform_params)

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
        # Create global_step.
        with tf.device(deploy_config.variables_device()):
            global_step = tf.get_variable(
                    'global_step', dtype=tf.int32,
                    initializer=tf.constant_initializer(0),
                    trainable=False)

        deployed_model = model_deploy.deploy(
                config=deploy_config, model_fn=network.clone_function(),
                args=None, kwargs=None)

        # ===

        # ======== instantiate clones
        clones = model_deploy.create_clones(
                deploy_config, network.clone_function)

        # model_deploy.optimize_clones(clones, network.optimizer)
        total_loss, clones_gradients = model_deploy.optimize_clones(
                clones,
                network.optimizer,
                var_list=variables_to_train)

        for epoch in range(FLAGS.num_epochs):
            # ========= Training
            t = 0
            shuffle(files_train)
            sess.run(iterator.initializer,
                     feed_dict={filenames_placeholder: files_train,
                                network.is_training:   True
                                })
            train_loss = []
            next_batch = iterator.get_next()

            while True:
                try:
                    input_prot_out = sess.run(next_batch)
                except tf.errors.OutOfRangeError:
                    break


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


def main(_):
    run_training_multigpu()


if __name__ == '__main__':
    tf.app.run()
