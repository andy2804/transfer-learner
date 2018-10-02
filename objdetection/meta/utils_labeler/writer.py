import os

import numpy as np
from PIL import Image


class Writer:
    def __init__(self, output_folder):
        if os.path.isdir(output_folder):
            self._out = output_folder
        else:
            try:
                os.mkdir(output_folder)
                self._out = output_folder
            except IOError as e:
                e += ["Error attempting to create %s folder" % output_folder]

    def dump(self, events_im, rgb_im, source_id, events_enc):
        """
        Save as png
        :param events_im:
        :param rgb_im:
        :param source_id:
        :param events_enc:
        :return:
        """
        events_im = np.stack([((events_im + 1) * 255 / 2).astype(np.uint8)] * 3, axis=-1)
        events_im = Image.fromarray(events_im, 'RGB')
        rgb_im = Image.fromarray(rgb_im, 'RGB')
        events_im_file = os.path.join(self._out, source_id + "_events" + events_enc)
        rgb_im_file = os.path.join(self._out, source_id + "_rgb")
        events_im.save(events_im_file + ".png", "PNG")
        rgb_im.save(rgb_im_file + ".png", "PNG")
