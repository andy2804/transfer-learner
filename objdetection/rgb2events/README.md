# rgb2events TFRecords Builder

This package contains all files needed to create a dataset in the IR 
operating domain from RGB images using semi-supervised learning. 

## Parameters

- `dataset_dir` : contains the path to the images to be evaluated and encoded
- `labels_net` : specifies the label set to be used with the object detector
- `labels_out` : specifies the output labels to be converted to
- `net_arch` : specify which object detector architecture to use, also see `objdet_frozengraph.py`

---

- `weight_fn` : weighting function to be used for decay of events in the image frame
- `time_window` : time window over which to accumulate events in [ms]
- `tl_events_time_window` : time window for the learning filter
- `tl_keepthresh` : lower score threshold to keep a detection pair
- `tl_min_img_peri` : minimum perimeter of the bounding box of detected objects in pixels

---

- `debug` : for debugging purposes only, shows the input images and the object detected
- `generate_plots` : whether or not to create a plot showing statistics of the encoding process

## Learning Filter

This section defines the learning filter for the rgb2events converter, 
also see `learning_filter_tfrecord.py`
The purpose of the learning filter is to decide, whether a detected object in 
one domain is also visible in the other operating domain such as RGB and Events in this case.

For the rgb2events case, it calculates an observability score, which is a weighted function consisting
of an activity score and overlap score. These are defined as follows:

### Activity score
    # Activity Score
    activity_score = 1 + (abs(1 - (np.sum(events_crop) / np.sum(rgb_crop))) * -1)
    
Where as `events_crop` and `rgb_crop` represent normalized gradient images of the cropped corresponding frames.
    
### Overlap score
    # Overlap score
    overlap_score = np.sum(np.multiply(img_box, img_events_box)) / np.sum(img_box)
    
Where as `img_events_box` and `img_box` represent normalized gradient images of the cropped corresponding frames.
    
### Observability score
    # Weighting
    score = min(max((activity_score + 2 * overlap_score) / 3.0, 0.0), 1.0)
    
Note that if the overall score is too high, the score is decreased again to compensate for frames
which contain alot of events due light artifacts such as laterns at night, which may create halos of events.

On top of that, a cutoff threshold can be defined, that automatically rejects object with a
bounding box perimeter smaller than the specified value. This helps to remove arbitrary detections.

## Run the rgb2events pipeline

1. First create the dataset using `run_build_tfrecords.py` with the preferred parameters.