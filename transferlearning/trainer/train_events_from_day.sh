#!/usr/bin/env bash

cd $HOME/models/research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Job_name can't have spaces and should match the ending of the config file
job_name=events_lf2_v2_ext

# Actual call
CUDA_VISIBLE_DEVICES=4,5,6 python3 object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=$HOME/WormholeLearning/resources/nets_cfgs/configs/ssd_inceptionv2_$job_name.config \
    --train_dir=/media/sdc/andya/wormhole_learning/models/obj_detector_retrain/train_$job_name/ \
    --num_clones 3