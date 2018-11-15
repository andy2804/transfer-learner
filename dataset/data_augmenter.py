"""
author: az
"""
from abc import ABC, abstractmethod

from objdetection.meta.utils_generic.magic_constants import \
    (DataAugmentationParameters, DEFAULT_DATAAUG_PARAMS)


class DataAugmenter(ABC):
    def __init__(self, params):
        self._dataaug_par = None
        self.dataaug_par = params

    @abstractmethod
    def augment(self, input_proto):
        """
        Builds the chain of data augmentation
            :param input_proto:{key_of_dict image:
                                key_of_dict events:
                                key_of_dict xcent:
                                key_of_dict ycent:
                                key_of_dict w:
                                key_of_dict h:}
            :returns : updates dictionary in place
        """
        pass

    @property
    def dataaug_par(self):
        return self._dataaug_par

    @dataaug_par.setter
    def dataaug_par(self, params):
        if isinstance(params, DataAugmentationParameters):
            self._dataaug_par = params
        else:
            print("Invalid data augmentation parameters, values set to default!")
            self._dataaug_par = DEFAULT_DATAAUG_PARAMS
