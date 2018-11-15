"""
author: aa & az
It contains almost all the parameters structures and  default values
Notes on options & parameters:

Center offset, width and height are relative to the unitary cell of the 
feature map
    featmap_layers= {"name": {'size': (width, height),
                            'dbox':[(Xcenter_offset, Ycenter_offset, width, 
                            height),(...)]}}
"""
from collections import namedtuple, OrderedDict

# Cameras details
Camera = namedtuple('Camera', ['width',
                               'height',
                               'channels'])
DAVIS240c = Camera(width=240,
                   height=180,
                   channels=1)

SILICON_EYE = Camera(width=326,
                     height=260,
                     channels=1)

PYLONrgb = Camera(width=2592,
                  height=2048,
                  channels=3)

BOSON = Camera(width=640,
               height=512,
               channels=1)

# todo is the rest still needed?

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

test_aps = OrderedDict({1: 0.0044695697919031,
                        2: 0.011939056853986994,
                        3: 0.15135318481642793,
                        4: 1.0043452053736862e-06,
                        5: 0.1281291667923279,
                        6: 3.904870075861212e-07,
                        7: 0.5729061399345241
                        })

test_corestats = OrderedDict({0.0: {'acc':  {1: 0.0014352245101197972,
                                             2: 0.0019000044039837178,
                                             3: 0.03596053616085557,
                                             4: 0.00012849341471249598,
                                             5: 0.008968535486729648,
                                             6: 1.2235259571031799e-05,
                                             7: 0.11583635298559936
                                             },
                                    'fp':   {1: 204552,
                                             2: 317290,
                                             3: 526970,
                                             4: 124504,
                                             5: 418246,
                                             6: 245190,
                                             7: 934286
                                             },
                                    'n_gt': {1: 441, 2: 1082, 3: 39539, 4: 2047, 5: 2791, 6: 94,
                                             7: 123834
                                             },
                                    'rec':  {1: 0.6666666666666666,
                                             2: 0.5582255083179297,
                                             3: 0.4971547080098131,
                                             4: 0.007816316560820713,
                                             5: 1.35614475098531,
                                             6: 0.031914893617021274,
                                             7: 0.9884442075681962
                                             },
                                    'tp':   {1: 294, 2: 604, 3: 19657, 4: 16, 5: 3785, 6: 3,
                                             7: 122403
                                             }
                                    },
                              0.1: {'acc':  {1: 0.04590163934426229,
                                             2: 0.04711425206124853,
                                             3: 0.4602433979508394,
                                             4: 0,
                                             5: 0.446360153256705,
                                             6: 0.0,
                                             7: 0.594136849245379
                                             },
                                    'fp':   {1: 582, 2: 2427, 3: 11221, 4: 0, 5: 578, 6: 89,
                                             7: 57441
                                             },
                                    'n_gt': {1: 441, 2: 1082, 3: 39539, 4: 2047, 5: 2791, 6: 94,
                                             7: 123834
                                             },
                                    'rec':  {1: 0.06349206349206349,
                                             2: 0.11090573012939002,
                                             3: 0.24198892232985153,
                                             4: 0.0,
                                             5: 0.1669652454317449,
                                             6: 0.0,
                                             7: 0.6790299917631668
                                             },
                                    'tp':   {1: 28, 2: 120, 3: 9568, 4: 0, 5: 466, 6: 0, 7: 84087}
                                    },
                              0.2: {'acc':  {1: 0.049773755656108594,
                                             2: 0.06292286874154263,
                                             3: 0.5150106527212861,
                                             4: 0,
                                             5: 0.602112676056338,
                                             6: 0.0,
                                             7: 0.6477055764039765
                                             },
                                    'fp':   {1: 210, 2: 1385, 3: 7512, 4: 0, 5: 226, 6: 28, 7: 42808
                                             },
                                    'n_gt': {1: 441, 2: 1082, 3: 39539, 4: 2047, 5: 2791, 6: 94,
                                             7: 123834
                                             },
                                    'rec':  {1: 0.024943310657596373,
                                             2: 0.08595194085027727,
                                             3: 0.2017501707175194,
                                             4: 0.0,
                                             5: 0.12253672518810463,
                                             6: 0.0,
                                             7: 0.6355605084225657
                                             },
                                    'tp':   {1: 11, 2: 93, 3: 7977, 4: 0, 5: 342, 6: 0, 7: 78704}
                                    },
                              0.3: {'acc':  {1: 0.08411214953271028,
                                             2: 0.08221993833504625,
                                             3: 0.5492946521080736,
                                             4: 0,
                                             5: 0.7475,
                                             6: 0.0,
                                             7: 0.6819058361228351
                                             },
                                    'fp':   {1: 98, 2: 893, 3: 5655, 4: 0, 5: 101, 6: 17, 7: 34970},
                                    'n_gt': {1: 441, 2: 1082, 3: 39539, 4: 2047, 5: 2791, 6: 94,
                                             7: 123834
                                             },
                                    'rec':  {1: 0.02040816326530612,
                                             2: 0.07393715341959335,
                                             3: 0.17430891018993905,
                                             4: 0.0,
                                             5: 0.10713006091006808,
                                             6: 0.0,
                                             7: 0.6053749374162185
                                             },
                                    'tp':   {1: 9, 2: 80, 3: 6892, 4: 0, 5: 299, 6: 0, 7: 74966}
                                    },
                              0.4: {'acc':  {1: 0.07547169811320754,
                                             2: 0.0899854862119013,
                                             3: 0.5719472577871202,
                                             4: 0,
                                             5: 0.7894736842105263,
                                             6: 0.0,
                                             7: 0.7091923380726698
                                             },
                                    'fp':   {1: 49, 2: 627, 3: 4480, 4: 0, 5: 72, 6: 9, 7: 29453},
                                    'n_gt': {1: 441, 2: 1082, 3: 39539, 4: 2047, 5: 2791, 6: 94,
                                             7: 123834
                                             },
                                    'rec':  {1: 0.009070294784580499,
                                             2: 0.05730129390018484,
                                             3: 0.1513948253623005,
                                             4: 0.0,
                                             5: 0.09673951988534575,
                                             6: 0.0,
                                             7: 0.580026487071402
                                             },
                                    'tp':   {1: 4, 2: 62, 3: 5986, 4: 0, 5: 270, 6: 0, 7: 71827}
                                    },
                              0.5: {'acc':  {1: 0.06896551724137931,
                                             2: 0.10236220472440945,
                                             3: 0.5973921994738648,
                                             4: 0,
                                             5: 0.7966101694915254,
                                             6: 0.0,
                                             7: 0.7323439918859767
                                             },
                                    'fp':   {1: 27, 2: 456, 3: 3520, 4: 0, 5: 60, 6: 6, 7: 25070},
                                    'n_gt': {1: 441, 2: 1082, 3: 39539, 4: 2047, 5: 2791, 6: 94,
                                             7: 123834
                                             },
                                    'rec':  {1: 0.0045351473922902496,
                                             2: 0.04805914972273567,
                                             3: 0.132097422797744,
                                             4: 0.0,
                                             5: 0.08419921175206019,
                                             6: 0.0,
                                             7: 0.553927031348418
                                             },
                                    'tp':   {1: 2, 2: 52, 3: 5223, 4: 0, 5: 235, 6: 0, 7: 68595}
                                    },
                              0.6: {'acc':  {1: 0.0,
                                             2: 0.129973474801061,
                                             3: 0.6175522220664517,
                                             4: 0,
                                             5: 0.8110236220472441,
                                             6: 0.0,
                                             7: 0.754942773983325
                                             },
                                    'fp':   {1: 16, 2: 328, 3: 2728, 4: 0, 5: 48, 6: 3, 7: 21133},
                                    'n_gt': {1: 441, 2: 1082, 3: 39539, 4: 2047, 5: 2791, 6: 94,
                                             7: 123834
                                             },
                                    'rec':  {1: 0.0,
                                             2: 0.04528650646950092,
                                             3: 0.11140898859354055,
                                             4: 0.0,
                                             5: 0.07380867072733788,
                                             6: 0.0,
                                             7: 0.5257360660238707
                                             },
                                    'tp':   {1: 0, 2: 49, 3: 4405, 4: 0, 5: 206, 6: 0, 7: 65104}
                                    },
                              0.7: {'acc':  {1: 0.0,
                                             2: 0.12595419847328243,
                                             3: 0.6376991150442478,
                                             4: 0,
                                             5: 0.817351598173516,
                                             6: 0.0,
                                             7: 0.7807127347043045
                                             },
                                    'fp':   {1: 10, 2: 229, 3: 2047, 4: 0, 5: 40, 6: 2, 7: 17168},
                                    'n_gt': {1: 441, 2: 1082, 3: 39539, 4: 2047, 5: 2791, 6: 94,
                                             7: 123834
                                             },
                                    'rec':  {1: 0.0,
                                             2: 0.030499075785582256,
                                             3: 0.09112521813905258,
                                             4: 0.0,
                                             5: 0.06413471873880329,
                                             6: 0.0,
                                             7: 0.49358011531566454
                                             },
                                    'tp':   {1: 0, 2: 33, 3: 3603, 4: 0, 5: 179, 6: 0, 7: 61122}
                                    },
                              0.8: {'acc':  {1: 0.0,
                                             2: 0.16233766233766234,
                                             3: 0.6624513618677043,
                                             4: 0,
                                             5: 0.8342857142857143,
                                             6: 0,
                                             7: 0.8082643427741467
                                             },
                                    'fp':   {1: 2, 2: 129, 3: 1388, 4: 0, 5: 29, 6: 0, 7: 13201},
                                    'n_gt': {1: 441, 2: 1082, 3: 39539, 4: 2047, 5: 2791, 6: 94,
                                             7: 123834
                                             },
                                    'rec':  {1: 0.0,
                                             2: 0.02310536044362292,
                                             3: 0.0688940033890589,
                                             4: 0.0,
                                             5: 0.052310999641705484,
                                             6: 0.0,
                                             7: 0.4493838525768367
                                             },
                                    'tp':   {1: 0, 2: 25, 3: 2724, 4: 0, 5: 146, 6: 0, 7: 55649}
                                    },
                              0.9: {'acc':  {1: 0,
                                             2: 0.16923076923076924,
                                             3: 0.6928539724811362,
                                             4: 0,
                                             5: 0.8677685950413223,
                                             6: 0,
                                             7: 0.8429092851419924
                                             },
                                    'fp':   {1: 0, 2: 54, 3: 692, 4: 0, 5: 16, 6: 0, 7: 8823},
                                    'n_gt': {1: 441, 2: 1082, 3: 39539, 4: 2047, 5: 2791, 6: 94,
                                             7: 123834
                                             },
                                    'rec':  {1: 0.0,
                                             2: 0.010166358595194085,
                                             3: 0.03948000708161562,
                                             4: 0.0,
                                             5: 0.03762092439985668,
                                             6: 0.0,
                                             7: 0.3823021141205162
                                             },
                                    'tp':   {1: 0, 2: 11, 3: 1561, 4: 0, 5: 105, 6: 0, 7: 47342}
                                    },
                              1.0: {'acc':  {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0
                                             },
                                    'fp':   {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
                                    'n_gt': {1: 441, 2: 1082, 3: 39539, 4: 2047, 5: 2791, 6: 94,
                                             7: 123834
                                             },
                                    'rec':  {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0
                                             },
                                    'tp':   {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
                                    }
                              })
