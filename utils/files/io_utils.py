import os

import yaml
from PIL import Image

__all__ = ["read_filenames"]

ROOT_DIR = "WormholeLearning/"
NETS_CKPT_DIR = "resources/nets_ckpt/"
NETS_CKPT_PATH = os.path.join(os.getcwd()[:os.getcwd().index(ROOT_DIR)], ROOT_DIR, NETS_CKPT_DIR)


def read_filenames(dir, filter_keywords, main_sensor_name, aux_sensor_name, filetype='png'):
    """
    Reads filenames for any arbitrary sensor combinations specified in the function call.
    Additionally the filetype can be specified. The image pair require to have the same name.
    :param dir:
    :return:
    """
    num = lambda x: int("".join(filter(str.isdigit, os.path.basename(x))))
    if os.path.exists(dir):
        main_sensor_images = [os.path.join(path_tuple[0], file) for path_tuple in os.walk(dir) for
                              file in path_tuple[2] if main_sensor_name in file and any(
                    [s in path_tuple[0] for s in filter_keywords]) and filetype in file]
        main_sensor_images.sort()
        if aux_sensor_name is None:
            return main_sensor_images
        aux_sensor_images = [os.path.join(path_tuple[0], file) for path_tuple in os.walk(dir) for
                             file in path_tuple[2] if aux_sensor_name in file and any(
                    [s in path_tuple[0] for s in filter_keywords]) and filetype in file]
        aux_sensor_images.sort()

        # Sanity checks to make sure they have been read out with the same order
        assert len(aux_sensor_images) == len(main_sensor_images)
        cond = [num(aux_sensor_images[i]) == num(main_sensor_images[i]) for i in
                range(len(main_sensor_images))]
        assert all(cond)
        return list(zip(main_sensor_images, aux_sensor_images))
    else:
        raise IOError("\tThe requested directory does not exists")


def list_files(input_dir, filter):
    if os.path.isdir(input_dir):
        files = [os.path.join(path_tuple[0], file) for path_tuple in os.walk(input_dir) for
                 file in path_tuple[2] if all([s in file for s in filter]) and FILE_TYPE in file]
        return files


def load_dict_from_yaml(yaml_file):
    with open(yaml_file, "r") as fs:
        yaml_dict = yaml.load(fs)
    return yaml_dict


def load_arch_dict(config_file):
    """
    Load labels from json file
    :param json_file:
    :return:
    """
    config_file = os.path.join(NETS_CKPT_PATH, config_file + ".yaml")
    with open(config_file, "r") as fs:
        arch_dict = yaml.load(fs)
    return arch_dict


def export_transfer_step_img(img_array, output_dir, count):
    """
    Exports verbose images from transfer step for visual inspection
    :param img_array:
    :param output_dir:
    :param output_file:
    :return:
    """

    # Check if output path exists
    if os.path.isdir(output_dir):
        pass
    else:
        try:
            os.makedirs(output_dir)
        except IOError as e:
            e += ["Error attempting to create %s folder" % output_dir]

    # Save image as png
    img = Image.fromarray(img_array, 'RGB')
    output_file = os.path.join(output_dir, "%06d.jpg" % count)
    img.save(output_file, "JPEG", quality=90)
