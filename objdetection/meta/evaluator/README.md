## Evaluation

### rgb baseline
```
CUDA_VISIBLE_DEVICES=5 python3 object_detection/eval.py \
--logtostderr \
--pipeline_config_path=/home/azanardi/zauron/events_objdetector/trainer/configs/ssd_inception_v2_zauronscapes.config \ 
--checkpoint_dir=/shared_experiments/training/train/ckpt \ 
--eval_dir=/shared_experiments/testing/eval
```

### events obj-detector
```
CUDA_VISIBLE_DEVICES=5 python3 object_detection/eval.py \
--logtostderr \
--pipeline_config_path=/home/azanardi/zauron/events_objdetector/trainer/configs/ssd_inception_v2_zauronscapes.config \ 
--checkpoint_dir=/home/ale/catkin_ws/src/zauron/resources/nets_ckpt/ssd_inception_v2_coco_11_06_2017/frozen_inference_graph.pb \ 
--eval_dir=/shared_experiments/testing/eval


```