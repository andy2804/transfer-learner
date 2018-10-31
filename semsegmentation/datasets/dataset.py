from abc import ABC


class Dataset(ABC):
    def __init__(self, labels, cmap):
        self._labels = labels
        self._label_colormap = cmap

    @property
    def colormap(self, ):
        return self._label_colormap

    @colormap.setter
    def colormap(self, cmap):
        try:
            assert len(cmap.shape) == 2
            assert cmap.shape[1] == 3
        except AssertionError:
            print("Invalid colormap, no changes have happened")

    @property
    def labels(self, ):
        return self._labels

    @labels.setter
    def labels(self, labels):
        try:
            assert len(labels.shape) == 1
            assert labels.shape[0] > 0
        except AssertionError:
            print("Invalid labels, no changes have happened")
