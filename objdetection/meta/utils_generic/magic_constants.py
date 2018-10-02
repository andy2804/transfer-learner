"""
author: az
It contains almost all the parameters structures and  default values
Notes on options & parameters:

Center offset, width and height are relative to the unitary cell of the 
feature map
    featmap_layers= {"name": {'size': (width, height),
                            'dbox':[(Xcenter_offset, Ycenter_offset, width, 
                            height),(...)]}}
"""
from collections import namedtuple

# Cameras details
Camera = namedtuple('Camera', ['width',
                               'height',
                               'channels'])
DAVIS240c = Camera(width=240,
                   height=180,
                   channels=1)

PYLONrgb = Camera(width=2592,
                  height=2048,
                  channels=3)

# ========== General options for network
""" Documentation #todo
'input_shape':
'input_format':
'num_classes':
'no_annotation_label':
'prior_scaling':
'featmap_layers': 
Center offset, width and height are relative to the unitary cell of the 
feature map
    featmap_layers= {"name": {'size': (width, height),
                            'dbox':[(Xcenter_offset, Ycenter_offset, width, 
                            height),(...)]}}
'featmap_scales_range':
'aspectratio_bias':
'iou_thresh':
'iou_neg_mining_thresh':
'optimizer_name':
'nms_iou_thresh':
'nms_maxN':
'gt_size':
'pd_size':
"""
# todo would be better instead of params and options to have
# structural parameters vs learning parameters
ObjDetParams = namedtuple('ObjDetParams', ['input_shape',
                                           'input_format',
                                           'num_classes',
                                           'nms_iou_thresh',
                                           'nms_maxN',
                                           'no_annotation_label',
                                           'retrival_confidence_thresh'
                                           ])
DEFAULT_OBJDET_PAR = ObjDetParams(
        # Non maximum suppression
        nms_iou_thresh=0.25,
        nms_maxN=15,
        input_format='HWC',
        input_shape=(None, DAVIS240c.height, DAVIS240c.width, DAVIS240c.channels),
        num_classes=1 + 7,
        no_annotation_label=0,
        # Retrival confidence
        retrival_confidence_thresh=.5
)
# ============== SSDNetwork parameters
SSDnetParams = namedtuple('SSDnetParams', [
    'prior_scaling',
    'featmap_layers',
    'featmap_scales_range',
    'aspectratio_bias',
    'iou_thresh',
    'iou_neg_mining_thresh',
    'neg_pos_ratio',
    'gt_size',
    'pd_size'
])

DEFAULT_SSDnet_PARAMS = SSDnetParams(
        iou_thresh=0.5,
        iou_neg_mining_thresh=0.5,
        prior_scaling=(0.1, 0.1, 0.2, 0.2),
        featmap_layers={
            # size (W,H), dbox [(xcent_offset, y_cent_offset, w_fmscaled,
            # h_fmscaled),..]
            'block4': {'size':         (19, 19),
                       'aspect_ratio': (1, 1, 2, 3, 1 / 2, 1 / 3)
                       },
            'block5': {'size':         (10, 10),
                       'aspect_ratio': (1, 1, 2, 3, 1 / 2, 1 / 3)
                       },
            'block6': {'size':         (5, 5),
                       'aspect_ratio': (1, 1, 2, 3, 1 / 2, 1 / 3)
                       },
            'block7': {'size':         (3, 3),
                       'aspect_ratio': (1, 1, 2, 3, 1 / 2, 1 / 3)
                       },
            'block8': {'size':         (1, 1),
                       'aspect_ratio': (1, 2, 3, 1 / 2, 1 / 3)
                       }
        },
        featmap_scales_range=(0.25, 0.925),
        aspectratio_bias=DAVIS240c.width / DAVIS240c.height,
        neg_pos_ratio=3,  # negative:positive
        gt_size=25,
        pd_size=100,
)

# ============== More training-specific hyper parameters
LearnParams = namedtuple('LearningParameters', ['optimizer_name',
                                                'weight_decay',
                                                'loc_loss_weight',
                                                'learning_rate',
                                                'minimizer_eps'])
DEFAULT_LEARN_PARAMS = LearnParams(
        # Options for optimization
        optimizer_name='Adam',
        minimizer_eps=1e-7,
        learning_rate=0.00005,
        weight_decay=0.0001,
        loc_loss_weight=3.,
)

# ============== Data augmentation parameters

DataAugmentationParameters = namedtuple("DataAugmentationParmeters",
                                        ["flipX_bool",
                                         "flip_polarity_bool",
                                         "random_quant_bool",
                                         "sample_distorted_bbox_bool",
                                         "sample_time_axis_bool",
                                         "random_yshift_bool"])
DEFAULT_DATAAUG_PARAMS = DataAugmentationParameters(
        flipX_bool=True,
        flip_polarity_bool=True,
        random_quant_bool=True,
        sample_distorted_bbox_bool=True,
        sample_time_axis_bool=True,
        random_yshift_bool=True,
)

# ============== Event transform parameters

EventTransformParameters = namedtuple('EventTransformParameters',
                                      ["weight_fn",
                                       "time_window"])

DEFAULT_EVENT_TRANSFORM_PARAMS = EventTransformParameters(
        weight_fn='gaus',
        time_window=40)
