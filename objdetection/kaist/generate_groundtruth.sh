#!/usr/bin/env bash

# Script


# TRAINING
python3 run_generate_groundtruth.py --dataset_dir "/shared_experiments/kaist/training/day" \
--network_model 6 --tfrecord_name_prefix "KAIST_TRAINING_DAY" \
--labels_net mscoco_label_map.json --labels_out kaist_label_map.json \
--cuda_visible_devices "0" --verbose=0 &


# TESTING
python3 run_generate_groundtruth.py --dataset_dir "/shared_experiments/kaist/testing/night" \
--network_model 6 --tfrecord_name_prefix "KAIST_TESTING_NIGHT" \
--labels_net mscoco_label_map.json --labels_out kaist_label_map.json \
--cuda_visible_devices "1" --verbose=0 &

python3 run_generate_groundtruth.py --dataset_dir "/shared_experiments/kaist/testing/day" \
--network_model 6 --tfrecord_name_prefix "KAIST_TESTING_DAY" \
--labels_net mscoco_label_map.json --labels_out kaist_label_map.json \
--cuda_visible_devices "2" --verbose=0 &
