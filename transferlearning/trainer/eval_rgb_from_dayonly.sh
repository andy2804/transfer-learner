#!/usr/bin/env bash

cd $HOME/models/research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Job_name can't have spaces and should match the ending of the config file
job_name=rgb_dayonly_rss

# Start evaluation job
CUDA_VISIBLE_DEVICES=3 python3 object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path=$HOME/WormholeLearning/resources/nets_cfgs/configs/ssd_inceptionv2_$job_name.config \
    --checkpoint_dir=/media/sdc/andya/wormhole_learning/models/obj_detector_retrain/train_$job_name/ \
    --eval_dir=/media/sdc/andya/wormhole_learning/models/obj_detector_retrain/train_$job_name/eval