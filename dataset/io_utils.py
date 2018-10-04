import os
from glob import glob

__all__ = ["read_filenames"]


def read_filenames(dir):
    """
    Reads filenames for rgb and ir
    :param dir:
    :return:
    """
    if os.path.exists(dir):
        rgbs = [file for path_tuple in os.walk(dir) for file in
                glob(os.path.join(path_tuple[0], '*.png')) if 'visible' in file]
        irs = [file for path_tuple in os.walk(dir) for file in
               glob(os.path.join(path_tuple[0], '*.png')) if 'lwir' in file]
        # sanity checks to make sure they have been read out with the same order
        assert len(irs) == len(rgbs)
        cond = [os.path.basename(irs[i]) == os.path.basename(rgbs[i]) for i in range(len(rgbs))]
        assert all(cond)
        return list(zip(rgbs, irs))
    else:
        raise IOError("\tThe requested directory does not exists")
