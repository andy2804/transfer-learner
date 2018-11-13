"""
author: az
"""
import argparse
import os
import sys
from time import time

from PIL import Image

sys.path.append(os.getcwd()[:os.getcwd().index('objdetection')])
from rosbag_converter.converter import RosbagConverter


class RosbagExtractorForLabeling(RosbagConverter):
    def __init__(self,
                 file_format,
                 args_fov_fitter="",
                 output_base_dir=None):
        super().__init__(topics_to_convert=args.topics_to_convert,
                         args_fov_fitter={'zauron_cfg': args_fov_fitter},
                         args_detection_graph={'arch': 2})
        self._file_format = file_format
        self.output_base_dir = output_base_dir

    def run(self, rosbag_path):
        """ Override run method
        :return:
        """
        self.rosbagpath = rosbag_path
        # Assigns and generates output directory
        extractor.output_dir = self.output_base_dir + "/" + self.get_rosbagname()
        self._init_rosbag()
        self._get_camera_info()

        while self._current_chunk < self._max_chunks:
            self._current_chunk += 1
            chunk = self._get_chunk(self._chunk_size)
            # self._fovfit_chunk(chunk)
            self._extract_images(chunk)

    def _generate_img_filename(self, ts, prefix="", suffix=""):
        bag_name = self.get_rosbagname()
        file_name = os.path.join(self.output_dir,
                                 prefix +
                                 bag_name +
                                 "_{:d}".format(ts.to_nsec()) +
                                 suffix +
                                 '.' + self._file_format)
        return file_name

    def _extract_images(self, chunk, topic_name='/pylon_rgb/image_raw'):
        images_filenames = [(Image.fromarray(instance[topic_name].image, 'RGB'),
                             self._generate_img_filename(
                                     ts=instance[topic_name].ts, suffix="_rgb"))
                            for instance in chunk]
        for i, (image, filename) in enumerate(images_filenames):
            self._print_status("Storing image ", i)
            # t = time()
            image = image.resize((250, 250))
            image.save(filename, self._file_format, compress_level=1)
            # print("Resize + saving took {:.2f} [s]".format(time() - t))

    def _generate_events_filename(self, ts, prefix=""):
        pass

    def _extract_events(self, ts, prefix=""):
        pass

    @property
    def file_format(self):
        return self._file_format

    @file_format.setter
    def file_format(self, value):
        self._file_format = value

    @staticmethod
    def retrieve_bag_paths(input_path):
        """
        :param input_path:
        :return:
        """
        patharray = []
        # check for file or directory
        if os.path.isfile(input_path):
            patharray.append(input_path)
        else:
            if os.path.isdir(input_path):
                for file in os.listdir(input_path):
                    if file.endswith(".bag"):
                        patharray.append(os.path.join(input_path, file))
                print('Input rosbags:')
                list(map(print, patharray))
            elif not os.path.isdir(input_path):
                print('\,The path "%s" is neither a file neither a directory' % input_path)
                raise IOError
        return patharray


if __name__ == '__main__':
    # Parse input arguments
    parser = argparse.ArgumentParser(
            description='Extract frames for hand labeling')
    parser.add_argument('--rosbags', type=str, default='/shared_experiments/tmp/rosbags/'
                                                       'ODN_WMW_E0_2018-06-26-22-34-26.bag',
                        help='Input rosbag file/directory')
    parser.add_argument('--output', type=str, required=False,
                        default="/shared_experiments/tmp/rosbags/extracted_images",
                        help='Output folder where extracted files are being stored')
    parser.add_argument('--output_format', type=str, default='png', required=False,
                        help='Output folder where extracted files are being stored')
    parser.add_argument('--fov_fitter_args', type=str, required=False,
                        default='/home/azanardi/zauron/zauron_eye/config'
                                '/calibration/Zauron_v0.3.yaml',
                        help='Argument needed to help run_conversion know the baseline rotation '
                             'matrix and translation vector.')
    parser.add_argument('--cuda_visible_devices', type=str, default="0,",
                        help='List of gpus to be used, the others will be masked', required=False)
    parser.add_argument('--topics_to_convert', type=list,
                        default=["/pylon_rgb/image_raw/compressed", ],
                        help='', required=False)
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    print("Before extractor")
    extractor = RosbagExtractorForLabeling(file_format=args.output_format,
                                           args_fov_fitter=args.fov_fitter_args,
                                           output_base_dir=args.output)
    print("rosbags to extract:", args.rosbags)
    for bag in extractor.retrieve_bag_paths(args.rosbags):
        extractor.run(bag)
