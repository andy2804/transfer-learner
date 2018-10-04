# rgb2ir TFRecords Builder

This package contains all files needed to create a dataset in the IR 
operating domain from RGB images using semi-supervised learning. 

## Parameters

- `dataset_dir` : contains the path to the images to be evaluated and encoded
- `labels_net` : specifies the label set to be used with the object detector
- `labels_out` : specifies the output labels to be converted to
- `net_arch` : specify which object detector architecture to use, also see `objdet_frozengraph.py`

---

- `normalize` : this parameter specifies, whether or not to normalize 
images to zero mean and unit variance
- `per_image_normalization` : whether to do perform normalization per single 
image or over the whole dataset
- `learning_filter` : whether or not to use the learning filter. The learning 
filter is defined in the next subsection

---

- `verbose` : for debugging purposes only, shows the input images and the object detected
- `generate_plots` : whether or not to create a plot showing statistics of the encoding process

## Learning Filter

This section defines the learning filter for the rgb2ir converter, 
also see `learning_filter_tfrecord.py`
The purpose of the learning filter is to decide, whether a detected object in 
one domain is also visible in the other operating domain such as RGB and IR in this case.

For the rgb2ir case, it calculates an observability score for the detected object in both
domains and determines if the object is rejected, depending on the set threshold.
Here it is calculating the shannon entropy of the image according to [this](www1.idc.ac.il/toky/imageProc-10/.../04_histogram_10.ppt) formula.

On top of that, a cutoff threshold can be defined, that automatically rejects object with a
bounding box perimeter smaller than the specified value. This helps to remove arbitrary detections.

## Run the rgb2ir pipeline

1. First create the dataset using `run_build_tfrecords.py` with the preferred parameters.
2. Run `train_ir_from_day.sh` shell script to launch training with the recently created dataset.
   Be sure to have set up tensorflow models correctly for retraining.
3. After training run `export_ir_from_day.sh` script to derivate the frozen inference graph from
   the model.