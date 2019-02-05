#!/usr/bin/env bash

# specify path to tensorflow models research directory
cd $HOME/models/research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Job_name can't have spaces and should match the ending of the config file
job_name=name_of_protobuf_config
CUDA_VISIBLE_DEVICES=0,1,2

# Start file with arguments of protobuf config ot be used and cuda visible devices!
# e.g. "./train_example.sh name_of_protobuf_config 0,1,2"

if [ $# -eq 0 ];
then
    job_name=$job_name
else
    job_name=$1
    CUDA_VISIBLE_DEVICES=$2
fi

# Start evaluation job
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python3 object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path=$HOME/WormholeLearning/resources/nets_cfgs/configs/$job_name.config \
    --checkpoint_dir=/media/sdc/andya/wormhole_learning/models/obj_detector_retrain/train_$job_name/ \
    --eval_dir=/media/sdc/andya/wormhole_learning/models/obj_detector_retrain/train_$job_name/eval
