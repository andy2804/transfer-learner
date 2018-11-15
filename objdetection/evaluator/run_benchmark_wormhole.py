#!/usr/bin/env python3
from functools import partial
from multiprocessing import Pool

import tensorflow as tf

from objdetection.meta.evaluator.run_eval_frozengraph import main

# Order for mutable args:
# test day | test night
dataset_dir = "/shared_experiments/kaist/"
src_dir = ["testing/day/", "testing/night/"]
output_dir = "/shared_experiments/kaist/results"
testfiles = ["KAIST_TESTING_DAY_ZAURONSCAPES_RGB.tfrecord",
             "KAIST_TESTING_NIGHT_ZAURONSCAPES_RGB.tfrecord"]
normalize_images = False
network_model = [-10, -10]
labels_net_arch = "zauron_label_map.json"
labels_output = "zauron_label_map.json"
cuda_visible_devices = ["0", "1"]
num_readers = 2
num_parallel_calls = 5
batch_size = 1
prefetch_buffer_factor = 5
verbose = False
testname = ["kaist_tasting_day", "kaist_testing_night"]

app_run = partial(tf.app.run,
                  main=main,
                  dataset_dir=dataset_dir,
                  output_dir=output_dir,
                  normalize_images=normalize_images,
                  labels_net_arch=labels_net_arch,
                  labels_output=labels_output,
                  num_readers=num_readers,
                  num_parallel_calls=num_parallel_calls,
                  batch_size=batch_size,
                  prefetch_buffer_factor=prefetch_buffer_factor,
                  verbose=verbose,
                  )
flags = tf.flags
if __name__ == '__main__':
    with Pool(2) as pool:
        pool.map(app_run,
                 zip(src_dir, testfiles, network_model, cuda_visible_devices, testfiles))
