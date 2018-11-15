#!/usr/bin/env bash
# Script
########## Daytime only ##########
# python3 run_eval_frozengraph.py \
--dataset_dir /shared_experiments/kaist/ \
--src_dir testing/day/ \
--output_dir /shared_experiments/kaist/results \
--network_model -4 \
--testfiles "KAIST_TESTING_DAY_rgb.tfrecord" \
--testname 1_kaist_day_rgb \
--cuda_visible_devices "0" \
--labels_net_arch kaist_label_map.json \
--verbose=0 &

python3 run_eval_frozengraph.py \
--dataset_dir /shared_experiments/kaist/ \
--src_dir testing/night/ --output_dir /shared_experiments/kaist/results \
--network_model -4 \
--testfiles "KAIST_TESTING_NIGHT_rgb_handlabeled.tfrecord" \
--testname 1_kaist_night_rgb_nodiff \
--cuda_visible_devices "5" \
--labels_net_arch kaist_label_map.json \
--verbose=0 \
--eval_difficult=0 &

######### IR ##########
# python3 run_eval_frozengraph.py \
--dataset_dir /shared_experiments/kaist/ \
--src_dir testing/day/ \
--output_dir /shared_experiments/kaist/results \
--network_model -5 \
--testfiles "KAIST_TESTING_DAY_ir.tfrecord" \
--testname 2_kaist_day_ir \
--cuda_visible_devices "1" \
--verbose=0 &

python3 run_eval_frozengraph.py \
--dataset_dir /shared_experiments/kaist/ \
--src_dir testing/night/ \
--output_dir /shared_experiments/kaist/results \
--network_model -5 \
--testfiles "KAIST_TESTING_NIGHT_ir_handlabeled.tfrecord" \
--testname 2_kaist_night_ir_nodiff \
--cuda_visible_devices "6" \
--verbose=0 \
--eval_difficult=0 &

########## Day and night ##########
# python3 run_eval_frozengraph.py \
--dataset_dir /shared_experiments/kaist/ \
--src_dir testing/day/ \
--output_dir /shared_experiments/kaist/results \
--network_model -11 \
--testfiles "KAIST_TESTING_DAY_rgb.tfrecord" \
--testname 3_kaist_day_rgb \
--cuda_visible_devices "3" \
--verbose=0 &

python3 run_eval_frozengraph.py \
--dataset_dir /shared_experiments/kaist/ \
--src_dir testing/night/ \
--output_dir /shared_experiments/kaist/results \
--network_model -11 \
--testfiles "KAIST_TESTING_NIGHT_rgb_handlabeled.tfrecord" \
--testname 3_kaist_night_rgb_nodiff \
--cuda_visible_devices "7" \
--verbose=0 \
--eval_difficult=0