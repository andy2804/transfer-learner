## Wormhole with kaist dataset
* Kaist dataset can be found [here](https://sites.google.com/site/pedestrianbenchmark/).

1. Train rgb detector only at daytime [see](../objdetection/rgb/README.md) 
2. Train ir detector from detector trained at point 1. All labels for training the ir detector are automatically 
generated in a semi-supervised fashion [see](../objdetection/rgb2ir/README.md)
3. Re-train rgb detector from detector trained at point 1. As data they will be used  
generated in a semi-supervised fashion [see](../objdetection/rgb2ir/README.md)

* Evaluation **# doc todo**

#### Folder structure for files and outputs (DEPRECATED)

```
/shared_experiments
+-- training
    +-- rosbags
        +-- *.bag and *.broken (run_health_check.py target)
        +-- converted_rosbags (output of run_conversion.py)
    +-- train
        +-- train_events_gaus80_lf14.tfrecord (output of run_convert_tfrecords.py, target forlder for training)
        +-- train_events_exp20_lf20.tfrecord
        +-- train_rgb_darkness.tfrecord
        +-- ckpt (todo diversify for each training)
+-- testing
    +-- rosbags
    +-- test_1
    +-- test_2
```