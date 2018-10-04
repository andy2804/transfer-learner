import os
from glob import glob

__all__ = ["read_filenames"]


def read_filenames(dir, main_sensor_name, aux_sensor_name, filetype='png'):
    """
    Reads filenames for any arbitrary sensor combinations specified in the function call.
    Additionally the filetype can be specified. The image pair require to have the same name.
    :param dir:
    :return:
    """
    if os.path.exists(dir):
        main_sensor_images = [file for path_tuple in os.walk(dir) for file in
                              glob(os.path.join(path_tuple[0], '*.' + filetype)) if
                              main_sensor_name in file]
        aux_sensor_images = [file for path_tuple in os.walk(dir) for file in
                             glob(os.path.join(path_tuple[0], '*.' + filetype)) if
                             aux_sensor_name in file]

        # Sanity checks to make sure they have been read out with the same order
        assert len(aux_sensor_images) == len(main_sensor_images)
        cond = [os.path.basename(aux_sensor_images[i]) == os.path.basename(main_sensor_images[i])
                for i in range(len(main_sensor_images))]
        assert all(cond)
        return list(zip(main_sensor_images, aux_sensor_images))
    else:
        raise IOError("\tThe requested directory does not exists")
