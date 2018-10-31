import numpy as np

# todo refactor as calss
from semsegmentation.datasets.dataset import Dataset


class Cityscapes(Dataset):

    def __init__(self):
        super(Cityscapes, self).__init__(
                self._get_default_labels(),
                self._get_default_colormap()
        )

    @staticmethod
    def _get_default_colormap():
        return np.array([
            [128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [102, 102, 156],
            [190, 153, 153],
            [153, 153, 153],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
            [152, 251, 152],
            [70, 130, 180],
            [220, 20, 60],
            [255, 0, 0],
            [0, 0, 142],
            [0, 0, 70],
            [0, 60, 100],
            [0, 80, 100],
            [0, 0, 230],
            [119, 11, 32],
        ])

    @staticmethod
    def _get_default_labels():
        return np.asarray(["road", "sidewalk",
                           "person", "rider",
                           "car", "truck", "bus", "on rails", "motorcycle", " bicycle",
                           "building", " wall", " fence",
                           "pole", "traffic sign", "traffic light",
                           "vegetation", " terrain",
                           "sky"])
