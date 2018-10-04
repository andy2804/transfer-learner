"""
author: az
"""

import json
import os
import sys
import tarfile
import time

import numpy as np
import six.moves.urllib as urllib
import tensorflow as tf
from recordclass import recordclass

ObjectDetected = recordclass('ObjectDetected', ['source', 'boxes', 'scores', 'classes', 'ts'])

# ======================
# Negative number indexes for networks trained by us. Positive numbers for nets trained by third
# parties.
# Each name must contain info about the net architecture, the dataset it has been trained on and
# the label map
ROOT_DIR = "zauron/"
ARCH_DICT = {
    # -1x: nets trained at daytime only, -2x: ir nets, -3x: retrained nets with daytime + ir labels
    -42: "ssd_inception_v2_kaist_ir035_rgb050_nightonly_RGB",
    -41: "ssd_inception_v2_kaist_ir035_rgb035_nightonly_RGB",
    -40: "ssd_inception_v2_kaist_ir035_rgb015_nightonly_RGB",
    -34: "ssd_inception_v2_kaist_ir035_rgb065_RGB",
    -33: "ssd_inception_v2_kaist_ir035_rgb050_RGB",
    -32: "ssd_inception_v2_kaist_ir035_rgb035_RGB",
    -31: "ssd_inception_v2_kaist_ir035_rgb015_RGB",
    -30: "ssd_inception_v2_kaist_ir050_rgb050_RGB",
    -24: "ssd_inception_v2_kaist_ir035_norm_IR",
    -23: "ssd_inception_v2_kaist_ir065_IR",
    -22: "ssd_inception_v2_kaist_ir050_IR",
    -21: "ssd_inception_v2_kaist_ir035_IR",
    -20: "ssd_inception_v2_kaist_ir015_IR",
    -10: "ssd_inception_v2_kaist_dayonly_RGB",
    1:   "ssd_mobilenet_v1_coco_11_06_2017",
    2:   "ssd_inception_v2_coco_2018_01_28",
    3:   "rfcn_resnet101_coco_11_06_2017",
    4:   "faster_rcnn_resnet101_coco_11_06_2017",
    5:   "faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28",
    6:   "faster_rcnn_nas_coco_2018_01_28",
}
NETS_CKPT_DIR = "resources/nets_ckpt/"
LABELS_DIR = "resources/labels/"


# =======================


class Detector:
    def __init__(self,
                 arch=2,
                 download_base='http://download.tensorflow.org/models/object_detection/',
                 labels_net_arch='mscoco_label_map.json',
                 labels_output=None,
                 retrieval_thresh=.5):
        """ Class obj detection from frozen graphs
        :type arch: int
        :type download_base: str
        :type labels_net_arch: str
        :param labels_net_arch: labels map proper of the network (the one it has been trained on)
        :type labels_output: str
        :param labels_output labels map desired by external user, if specified an extra remapping
        step from the labels_net_arch to labels_output is performed
        :type retrieval_thresh: float
        """
        assert arch in ARCH_DICT.keys()
        self._download_base = download_base
        self._model_name = ARCH_DICT[arch]
        self._model_file = ARCH_DICT[arch] + '.tar.gz'
        self._path_to_root = os.path.join(os.getcwd()[:os.getcwd().index(ROOT_DIR)], ROOT_DIR)

        # Path to frozen detection graph.
        # This is the actual model that is used for the object detection.
        self._path_to_ckpt_dir = os.path.join(self._path_to_root, NETS_CKPT_DIR)
        self._path_to_ckpt = os.path.join(self._path_to_ckpt_dir,
                                          self._model_name + '/frozen_inference_graph.pb')
        self._labels_netarch = labels_net_arch
        self._maybe_download()
        self._detection_graph, self._labels_netarch_dict = self._load_graph()
        self._config = tf.ConfigProto()
        self._config.gpu_options.allow_growth = True
        self._sess = tf.Session(graph=self._detection_graph, config=self._config)

        # Definite input and output Tensors for detection_graph
        self._image_tensor = self._detection_graph.get_tensor_by_name('image_tensor:0')
        self._detection_boxes = self._detection_graph.get_tensor_by_name('detection_boxes:0')
        self._detection_scores = self._detection_graph.get_tensor_by_name('detection_scores:0')
        self._detection_classes = self._detection_graph.get_tensor_by_name('detection_classes:0')
        self._num_detections = self._detection_graph.get_tensor_by_name('num_detections:0')
        self._retrival_thresh = retrieval_thresh

        # Labels transform
        self._labels_transform = None
        self._labels_output = labels_output
        if self._labels_output is not None:
            self._labels_output_dict = self._init_labels_transform(self._labels_output)

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

    def _load_labels(self, labels):
        """
        :type labels: str expected json file
        :return:
        """
        path_to_labels = os.path.join(self._path_to_root + LABELS_DIR, labels)
        try:
            with open(path_to_labels, "r") as f:
                raw_dict = json.load(f)
            # reformatting with key as int
            return {int(k): v for k, v in raw_dict.items()}
        except IOError as e:
            e.args += ("Error encountered trying to load {}".format(path_to_labels),)
            raise e

    def _load_graph(self):
        """
        Load frozen graph (arch) and its correspondent labels
        :return:
        """
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self._path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        labels_dict = self._load_labels(self._labels_netarch)
        return detection_graph, labels_dict

    def _retrieval_threshold_cutoff(self, classes, scores, boxes):
        idx_to_keep = np.where(np.greater_equal(np.squeeze(scores), self._retrival_thresh))
        boxes = np.squeeze(boxes)[idx_to_keep]
        classes = np.squeeze(classes.astype(int))[idx_to_keep]
        scores = np.squeeze(scores)[idx_to_keep]
        return classes, scores, boxes

    def _init_labels_transform(self, new_labels):
        """
        :type new_labels: str
        :return:
        """
        new_labels_dict = self._load_labels(new_labels)
        self._labels_transform = {}
        for label in self._labels_netarch_dict.values():
            new_value = next((new_label['id'] for new_label in new_labels_dict.values() if
                              label['name'] == new_label['name']), 0)
            transform = {label["id"]: new_value}
            self._labels_transform.update(transform)
        return new_labels_dict

    def remap_labels(self, objdetected, new_labels):
        """
        :param objdetected:
        :type new_labels: str path to .json file with new labels
        :return:
        """
        # todo deprecated, should be replaced by remap_labels_2
        if self._labels_transform is None:
            self._init_labels_transform(new_labels)
        if objdetected.classes.size != 0:
            objdetected.classes = np.vectorize(self._labels_transform.get)(objdetected.classes)
            idx_to_keep = np.where(objdetected.classes != 0)
            objdetected.classes = objdetected.classes[idx_to_keep]
            objdetected.scores = objdetected.scores[idx_to_keep]
            objdetected.boxes = objdetected.boxes[idx_to_keep]
        return objdetected

    def remap_labels_2(self, classes, scores, boxes):
        """
        :param classes:
        :param scores:
        :param boxes:
        :return:
        """
        # todo should replace remap_labels after appropriate testing
        if self._labels_transform is None:
            raise RuntimeError("Remapping labels requires labels_transform to be initialized")
        if classes.size != 0:
            classes = np.vectorize(self._labels_transform.get)(classes)
            # to support slicing without loosing dimensions
            idx_to_keep = list(zip(np.where(classes != 0)))
            classes = classes[idx_to_keep]
            scores = scores[idx_to_keep]
            boxes = boxes[idx_to_keep]
        return classes, scores, boxes

    def run_inference_on_img(self, image_in, source='detect_from_rgb'):
        """ Outputs the labels in the network architecture format
        :param image_in:
        :param source:
        :return ObjectDetected: boxes are Nx4 [y_min, x_min, y_max, x_max]
        """
        assert image_in.shape[2] == 3
        image_np_expanded = np.expand_dims(image_in, axis=0)
        # Actual detection.
        (boxes, scores, classes, _) = self._sess.run(
                [self._detection_boxes,
                 self._detection_scores,
                 self._detection_classes,
                 self._num_detections],
                feed_dict={self._image_tensor: image_np_expanded})
        # filter according to retrieval thresh
        classes, scores, boxes = self._retrieval_threshold_cutoff(classes, scores, boxes)
        return ObjectDetected(source=source,
                              boxes=boxes,
                              scores=scores,
                              classes=classes,
                              ts=0)

    @property
    def retrieval_thresh(self):
        return self._retrival_thresh

    @retrieval_thresh.setter
    def retrieval_thresh(self, value):
        if value < 0 or value > 1:
            raise ValueError("The retrieval threshold should be a number between 0 and 1.")
        else:
            self._retrival_thresh = value
            print("New retrieval threshold set to {0:.3f}".format(self._retrival_thresh))


if __name__ == '__main__':
    print("This is only for testing, do not run as main!")
    det_graph = Detector(arch=2)
    print("Test ended")
