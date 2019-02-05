"""
author: aa
"""

import os

import yaml

ROOT_DIR = "WormholeLearning/"
CFG_DIR = "transferlearning/config/"
CFG_PATH = os.path.join(os.getcwd()[:os.getcwd().index(ROOT_DIR)], ROOT_DIR, CFG_DIR)


def load_config(flags, config_file):
    """
    Load labels from json file
    :param json_file:
    :return:
    """
    config_file = os.path.join(CFG_PATH, config_file)
    with open(config_file, "r") as fs:
        config = yaml.load(fs)

    # ================ LABELS
    flags.DEFINE_string('labels_net', config['labels_net'], 'labels_net')
    flags.DEFINE_string('labels_out', config['labels_out'], 'labels_out')

    # ================ DIRECTORIES
    flags.DEFINE_string('dataset_dir', config['dataset_dir'], 'dataset_dir')

    # Check if a list of keywords or a dataset has been passed
    filter_keywords = config['filter_keywords']
    if not isinstance(config['filter_keywords'], str):
        subset_list = config['filter_subset']
        for index in subset_list:
            filter_keywords = filter_keywords[index]
    flags.DEFINE_list('filter_keywords', filter_keywords, 'filter_keywords')

    flags.DEFINE_string('main_sensor', config['main_sensor'], 'main_sensor')
    flags.DEFINE_string('aux_sensor', config['aux_sensor'], 'aux_sensor')
    flags.DEFINE_string('tfrecord_name_prefix', config['tfrecord_name_prefix'],
                        'tfrecord_name_prefix')
    flags.DEFINE_string('output_dir', config['output_dir'], 'output_dir')

    # ================ GPUS MASK
    flags.DEFINE_string('cuda_visible_devices', config['cuda_visible_devices'],
                        'cuda_visible_devices')

    # ================ SEMI-SUPERVISED LABEL GENERATOR
    flags.DEFINE_string('arch_config', 'zurich_rss_networks', 'arch_config')
    flags.DEFINE_integer('net_arch', config['net_arch'], 'net_arch')
    flags.DEFINE_float('retrieval_thresh', config['retrieval_thresh'], 'retrieval_thresh')

    # ================ PREPROCESSING
    flags.DEFINE_bool('normalize', config['normalize'], 'normalize')
    flags.DEFINE_bool('per_image_normalization', config['per_image_normalization'],
                      'per_image_normalization')
    flags.DEFINE_float('confidence_interval', config['confidence_interval'], 'confidence_interval')
    flags.DEFINE_bool('scale_back_using_cv2', config['scale_back_using_cv2'],
                      'scale_back_using_cv2')

    # ================ LEARNING FILTER
    flags.DEFINE_string('lf_mode', config['lf_mode'], 'lf_mode')
    flags.DEFINE_integer('min_obj_size', config['min_obj_size'], 'min_obj_size')
    flags.DEFINE_integer('lf_score_thresh', config['lf_score_thresh'], 'lf_score_thresh')
    flags.DEFINE_list('remove_roi', config['remove_roi'], 'remove_roi')
    flags.DEFINE_list('remove_shape', config['remove_shape'], 'remove_shape')

    # ================ VISUALIZATION AND PLOTS
    flags.DEFINE_string('verbose', config['verbose'], 'verbose')
    flags.DEFINE_bool('generate_plots', config['generate_plots'], 'generate_plots')
    flags.DEFINE_bool('show_plots', config['show_plots'], 'show_plots')
    flags.DEFINE_string('google_sheets', config['google_sheets'], 'google_sheets')

    return flags


def load_dict_from_yaml(yaml_file):
    with open(yaml_file, "r") as fs:
        yaml_dict = yaml.load(fs)
    return yaml_dict