import os
from random import shuffle

import tensorflow as tf

from objdetection.meta.datasets.encoder_tfrecord_googleapi import EncoderTFrecGoogleApi
from objdetection.meta.datasets.encoder_tfrecord_rosbag import EncoderTFrecRosbag
from objdetection.meta.utils_events.sae import Sae
from objdetection.meta.utils_generic import magic_constants
from objdetection.rgb2events.nets import ssd_common
from objdetection.rgb2events.tfrecords_builder.learning_filter_tfrecord import \
    TFRecordLearningFilter

__all__ = ['run_conversion']


class ConvertTFrecords(Sae):
    def __init__(self, events_transform_par):
        self._tfrecs_rosbag = EncoderTFrecRosbag()
        Sae.__init__(self, events_transform_par.weight_fn, events_transform_par.time_window)

    def parse_input(self, input_proto):
        """Input function to decode TFrecords and matching of dboxes.
        Data augmentation and matching with default boxes is performed.

        :param input_proto: single element of the batch for read in
        :returns : a dict of elements to feed the network.
        """
        input_proto_dict = self._tfrecs_rosbag.decode(input_proto)

        gt_labels, gt_boxes = ssd_common.format_gt(
                input_proto_dict["xcent"],
                input_proto_dict["ycent"],
                input_proto_dict["w"],
                input_proto_dict["h"],
                input_proto_dict["gt_labels"],
                gt_size=25)
        input_proto_dict["events"] = self._reshape_events(
                input_proto_dict["events"])
        return {"frame":     input_proto_dict["frame"],
                "frame_ts":  input_proto_dict["frame_ts"],
                "events":    input_proto_dict["events"],
                "gt_labels": gt_labels,
                "gt_boxes":  gt_boxes,
                "source_id": input_proto_dict["source_id"]
                }

    def _reshape_events(self, events, default_numbers=30000):
        """
        Fills or crop the events to default number to allow output stack into batches
        :param events:
        :return:
        """
        diff_len = default_numbers - tf.shape(events)[0]
        events = tf.cond(tf.greater(diff_len, 0),
                         true_fn=lambda: self._pad_zeros(events, diff_len),  # to few events
                         false_fn=lambda: events[-default_numbers:, :])  # to many events
        return events

    @staticmethod
    def _pad_zeros(_events, diff_len):
        pad = tf.zeros(
                tf.concat([[diff_len], tf.constant([4], dtype=tf.int32)], axis=0),
                dtype=tf.float64)
        return tf.concat([_events, pad], axis=0)


def run_conversion(flags):
    filenames = _get_filenames(flags)
    # Event Transform
    events_transform_par = magic_constants.EventTransformParameters(
            weight_fn=flags.weight_fn, time_window=flags.time_window)
    # Instantiate converter (needed to decode the full rosbag tfrecords)
    converter = ConvertTFrecords(events_transform_par)
    # Encoder
    encoder = EncoderTFrecGoogleApi()
    # Event Transform
    ev_transform = Sae(weight_fn=events_transform_par.weight_fn,
                       time_window=events_transform_par.time_window)
    # Learning_filter
    learning_filter = TFRecordLearningFilter(
            events_time_window=flags.tl_events_time_window,
            score_threshold=flags.tl_keepthresh,
            min_img_perimeter=flags.tl_min_img_peri,
            logstats=flags.generate_plots)
    with tf.Graph().as_default(), tf.Session() as sess:
        # Dataset Api
        filenames_placeholder = tf.placeholder(tf.string, shape=[None])
        dataset = tf.data.TFRecordDataset(
                filenames_placeholder, num_parallel_reads=flags.num_readers)
        dataset = dataset.map(converter.parse_input, num_parallel_calls=flags.num_parallel_calls)
        dataset = dataset.repeat(1)
        dataset = dataset.shuffle(buffer_size=flags.batch_size * flags.shuffle_buffer)
        dataset = dataset.batch(flags.batch_size)
        dataset = dataset.prefetch(buffer_size=flags.batch_size * flags.prefetch_buffer_factor)
        iterator = dataset.make_initializable_iterator()

        # ========= Visualising after decoding
        sess.run(iterator.initializer,
                 feed_dict={filenames_placeholder: filenames})
        next_batch = iterator.get_next()
        batch_count = 0
        out_file = _generate_filename(flags, ev_transform.get_encoding_str())
        with tf.python_io.TFRecordWriter(out_file) as writer:
            while True:
                try:
                    input_prot_out = sess.run(next_batch)
                    dynamic_batch_size = input_prot_out["frame_ts"].shape[0]

                    for i in range(dynamic_batch_size):
                        events_img = ev_transform.np_transform_events(
                                events=input_prot_out["events"][i, :],
                                to_ts=input_prot_out["frame_ts"][i, 0])
                        new_labels, new_boxes = learning_filter.apply_to_tfrec(
                                image=input_prot_out["frame"][i, :],
                                image_ts=input_prot_out["frame_ts"][i, 0],
                                events=input_prot_out["events"][i, :],
                                labels=input_prot_out["gt_labels"][i, :],
                                boxes=input_prot_out["gt_boxes"][i, :])
                        if flags.debug and batch_count > 30:
                            learning_filter.stats.make_plots()
                            # todo visualize decoded data for debug mode
                        else:
                            ex = encoder.encode({"boxes":  new_boxes,
                                                 "image":  events_img,
                                                 "labels": new_labels
                                                 })
                            writer.write(ex.SerializeToString())
                    batch_count += 1
                    print("\rProcessed batch data instances: {:d}".format(
                            batch_count * flags.batch_size), end='', flush=True)
                except tf.errors.OutOfRangeError:
                    if flags.generate_plots:
                        learning_filter.stats.make_plots()
                    learning_filter.stats.store_results()
                    break


def _get_filenames(flags):
    """
    Reading filenames of TFrecords and shuffling
    :return:
    """
    src_dir = os.path.join(flags.dataset_dir, flags.src_dir)
    filenames = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if
                 f.endswith(".tfrecord") and
                 all([banned not in f for banned in flags.black_list])]
    shuffle(filenames)
    return filenames


def _generate_filename(flags, events_encoding=''):
    """
    :return:
    """
    out_folder = os.path.join(flags.dataset_dir, flags.out_dir)
    name = flags.tfrecord_name + "_" + events_encoding + ".tfrecord"
    return os.path.join(out_folder, name)
