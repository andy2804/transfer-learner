import os

import numpy as np
import pandas as pd


class Loader:
    """
    Container of some methods to load events and frames stored with
    the RPG(Robotics PerceptionGroup) format
    """

    def __init__(self):
        pass

    @staticmethod
    def load_events(fold, events_file='events.txt'):
        path = os.path.join(fold, events_file)
        try:
            print("loading: ", fold)
            data_pd = pd.read_csv(path, sep=" ", header=None)
            data_pd.columns = ["ts", "x", "y", "polarity"]
            pos_bool = np.logical_and(data_pd.polarity, 1)
            pos = data_pd.values[pos_bool, :]
            neg = data_pd.values[np.logical_not(pos_bool), :]
        except pd.parser.CParserError:
            print("error loading: ", fold)
            raise IOError(fold)
        return {"pos": pos, "neg": neg}

    @staticmethod
    def _readTXT_frames(path):
        """
        :param path: struct of options, see above
        :return: dictionary of with key "time" and "name", values are the 
        correspondent lists
        """
        data_pd = pd.read_csv(path, sep=" ", header=None)
        data_pd.columns = ["ts", "frame_name"]
        return {"time": data_pd.ts, "name": data_pd.frame_name}

    @classmethod
    def load_frames(cls, fold):
        """
        Loads the images
        :param fold:
        :return: list of pairs [(image, timestamp), (image, timestamp),..]
        """
        path_txt_im = os.path.join(fold, "images.txt")
        dictFromTXT = cls._readTXT_frames(path_txt_im)
        images_list = dictFromTXT["name"].values
        ts_list = dictFromTXT["time"].values
        return list(zip(images_list, ts_list))
