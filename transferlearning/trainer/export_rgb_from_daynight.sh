#!/usr/bin/env bash

cd $HOME/models/research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Job_name can't have spaces
job_name=rgb_daynight_rss
dataset=zurich
ckpt_number=630629
gpu=3

# Pass arguments of checkpoint numbers to export several graphs at once!
# e.g. "./export_rgb_from_daynight.sh 230503 330981 612000"

if [ $# -eq 0 ]
then
ckpt=$ckpt_number
else
ckpt=$@
fi

for arg in $ckpt; do
    tag=$((arg / 1000))
    tag=${tag%.*}
    echo "Exporting graph ${job_name}_${arg}"

    # Actual call
    CUDA_VISIBLE_DEVICES=${gpu} python3 object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path $HOME/WormholeLearning/resources/nets_cfgs/configs/ssd_inceptionv2_${job_name}.config \
    --trained_checkpoint_prefix /media/sdc/andya/wormhole_learning/models/obj_detector_retrain/train_${job_name}/freeze/model.ckpt-${arg} \
    --output_directory /media/sdc/andya/wormhole_learning/models/obj_detector_retrain/ssd_inception_v2_${dataset}_${job_name}_${tag}/
done
