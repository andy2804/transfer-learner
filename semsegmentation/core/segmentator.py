import os
import sys
import tarfile
import time

import numpy as np
import six.moves.urllib as urllib
import tensorflow as tf
from PIL import Image
from matplotlib import gridspec
from matplotlib import pyplot as plt

ARCH_DICT = {
    0: "deeplabv3_mnv2_cityscapes_train",
}
ROOT_DIR = "WormholeLearning/"
NETS_CKPT_DIR = "resources/nets_ckpt/"
LABELS_DIR = "resources/labels/"
CITYSCAPES_LABELS = ("road", "sidewalk",
                     "person", "rider",
                     "car", "truck", "bus", "on rails", "motorcycle", " bicycle",
                     "building", " wall", " fence",
                     "pole", "traffic sign", "traffic light",
                     "vegetation", " terrain",
                     "sky")


class Segmentator:
    def __init__(self,
                 arch=0,
                 download_base='http://download.tensorflow.org/models/',
                 labels_net_arch='cityscape_map.json',
                 input_size=513):
        # core parameters
        self._input_size = input_size
        assert arch in ARCH_DICT.keys()
        self._download_base = download_base
        self._model_name = ARCH_DICT[arch]
        self._model_file = ARCH_DICT[arch] + '.tar.gz'
        self._path_to_root = os.path.join(os.getcwd()[:os.getcwd().index(ROOT_DIR)], ROOT_DIR)

        # Path to frozen detection graph.
        self._path_to_ckpt_dir = os.path.join(self._path_to_root, NETS_CKPT_DIR)
        self._path_to_ckpt = os.path.join(self._path_to_ckpt_dir,
                                          self._model_name + '/frozen_inference_graph.pb')
        self._labels_netarch = labels_net_arch
        self._maybe_download()

        # load graph and init input-output tensors
        self._segmentator_graph = self._load_graph()
        self._image_tensor = self._segmentator_graph.get_tensor_by_name('ImageTensor:0')
        self._semantic_pred_tensor = self._segmentator_graph.get_tensor_by_name(
                'SemanticPredictions:0')
        # init session
        self._config = tf.ConfigProto()
        self._config.gpu_options.allow_growth = True
        self._sess = tf.Session(graph=self._segmentator_graph, config=self._config)

    def _load_graph(self):
        """
        Load frozen graph (arch) and its correspondent labels
        :return:
        """
        segmentator_graph = tf.Graph()
        with segmentator_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self._path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return segmentator_graph

    def _maybe_download(self, ):
        if not os.path.exists(self._path_to_ckpt):
            try:
                print('Downloading neural network: %s' % self._model_name)
                opener = urllib.request.FancyURLopener()
                opener.retrieve(self._download_base + self._model_file,
                                filename=self._model_name,
                                reporthook=self._reporthook)
                tar_file = tarfile.open(self._model_name)
                for file in tar_file.getmembers():
                    file_name = os.path.basename(file.name)
                    if 'frozen_inference_graph.pb' in file_name:
                        tar_file.extract(file, self._path_to_ckpt_dir)
                try:
                    os.remove(self._model_name)
                except OSError:
                    pass
                print("\nNeural network downloaded!")
            except urllib.error.URLError as e:
                print("\nSomething went wrong connecting to tensorflow API, error:", e)
        else:
            print("Using pre-downloaded network.")

    @staticmethod
    def _reporthook(count, block_size, total_size):
        global start_time
        if count == 0:
            start_time = time.time()
            return
        curr_time = time.time()
        duration = curr_time - start_time
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024 * duration))
        percent = min(int(count * block_size * 100 / total_size), 100)
        sys.stdout.write("\r[%d%%] %d MB \tSpeed: %d MB/s" %
                         (percent, progress_size / (1024 * 1024), speed / 1e8))
        sys.stdout.flush()
        start_time = curr_time

    def run(self, image):
        """Runs inference on a single image.

            Args:
              image: A PIL.Image object, raw input image.

            Returns:
              resized_image: RGB image resized from original input image.
              seg_map: Segmentation map of `resized_image`.
            """
        width, height = image.size
        resize_ratio = 1.0 * self._input_size / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self._sess.run(
                self._semantic_pred_tensor,
                feed_dict={self._image_tensor: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map

    def vis_segmentation(image, seg_map):
        """Visualizes input image, segmentation map and overlay view."""
        plt.figure(figsize=(15, 5))
        grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

        plt.subplot(grid_spec[0])
        plt.imshow(image)
        plt.axis('off')
        plt.title('input image')

        plt.subplot(grid_spec[1])
        seg_image = label_to_color_image(seg_map).astype(np.uint8)
        plt.imshow(seg_image)
        plt.axis('off')
        plt.title('segmentation map')

        plt.subplot(grid_spec[2])
        plt.imshow(image)
        plt.imshow(seg_image, alpha=0.7)
        plt.axis('off')
        plt.title('segmentation overlay')

        unique_labels = np.unique(seg_map)
        ax = plt.subplot(grid_spec[3])
        plt.imshow(
                FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
        ax.yaxis.tick_right()
        plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
        plt.xticks([], [])
        ax.tick_params(width=0.0)
        plt.grid('off')
        plt.show()


if __name__ == '__main__':
    seg = Segmentator()
