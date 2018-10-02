## Training
### Google api:
If you want to rely on object detection api from google

* On Rudolf assuming deployment path of the zauron to be `/home/azanardi/`:

```
CUDA_VISIBLE_DEVICES=0,1,2 python3 object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=/home/azanardi/zauron/resources/nets_cfgs/configs/ssd_inceptionv2_day_n_night.config \
    --train_dir=/shared_experiments/kaist/models/obj_detector_retrain/models/train_rgb_day_n_night \
    --num_clones 3
```
### Our stuff
todo


## Evaluation




## Export graph

```
CUDA_VISIBLE_DEVICES=4 python3 object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path /home/azanardi/zauron/resources/nets_cfgs/configs/ssd_inceptionv2_day_n_night.config \
    --trained_checkpoint_prefix /shared_experiments/kaist/models/obj_detector_retrain/models/train_rgb_day_n_night/model.ckpt-75155 \
    --output_directory /shared_experiments/kaist/models/obj_detector_retrain/models/train_rgb_day_n_night/exported
```




## Retraining of the SSD Inception Net for IR

Run using following command:

```
CUDA_VISIBLE_DEVICES=5,6,7 python3 train.py \
    --logtostderr \
    --pipeline_config_path=ssd_inceptionv2_daytime_only.config \
    --train_dir=./models/train_0 \
    --num_clones 3
```

Export frozen graph using following command:

```
python3 export_inference_graph.py --input_type image_tensor --pipeline_config_path /home/andya/models/research/object_detection/obj_detector_daytime_only/ssd_inceptionv2_daytime_only.config --trained_checkpoint_prefix /home/andya/models/research/object_detection/obj_detector_daytime_only/models/train/model.ckpt-61668 --output_directory /home/andya/models/research/object_detection/obj_detector_daytime_only/fine_tuned_model
```