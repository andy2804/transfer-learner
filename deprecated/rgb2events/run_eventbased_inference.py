import json
import os
from random import sample

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from objdetection.meta.visualisation.static_helper import \
    visualize_boxes_and_labels_on_image_array

ROOT_DIR = "zauron/"
NETS_CKPT_DIR = "resources/nets_ckpt/"
LABELS_DIR = "resources/labels/"


class EventBasedObjDet:
    def __init__(self, path_to_ckpt, retrieval_thresh=.5, labels='zauronscapes_label_map.json'):
        self._labels_file = labels
        self._path_to_root = os.path.join(os.getcwd()[:os.getcwd().index(ROOT_DIR)], ROOT_DIR)

        self._path_to_ckpt = path_to_ckpt
        self._retrieval_thresh = retrieval_thresh

        self._detection_graph, self._labels_dict = self._load_graph()
        self._image_tensor = self._detection_graph.get_tensor_by_name('image_tensor:0')
        self._detection_boxes = self._detection_graph.get_tensor_by_name('detection_boxes:0')
        self._detection_scores = self._detection_graph.get_tensor_by_name('detection_scores:0')
        self._detection_classes = self._detection_graph.get_tensor_by_name('detection_classes:0')
        self._num_detections = self._detection_graph.get_tensor_by_name('num_detections:0')
        # sess
        self._config = tf.ConfigProto()
        self._config.gpu_options.allow_growth = True
        self._sess = tf.Session(graph=self._detection_graph, config=self._config)

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
        labels_dict = self._load_labels(self._labels_file)
        return detection_graph, labels_dict

    def run_inference_on_img(self, image_in, source='detect_from_rgb'):
        """
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
        boxes, scores, classes = self._retrival_threshold_cutoff(boxes, scores, classes)
        return boxes, scores, classes

    def visualize_rgb_detections(self, img, obj_detected, labels):
        """ Visualization of the results of a detection.
        :param img:
        :param obj_detected:
        :type obj_detected: ObjectDetected
        :return:
        """
        img_copy = np.copy(img)
        if labels is None:
            labels = self._labels_dict
        visualize_boxes_and_labels_on_image_array(
                img_copy,
                obj_detected.boxes,
                obj_detected.classes.astype(np.int32),
                obj_detected.scores,
                labels,
                use_normalized_coordinates=True,
                line_thickness=1)
        return img_copy

    def run_inference_on_img_and_plot(self, im):
        boxes, scores, classes = self.run_inference_on_img(im)
        visualize_boxes_and_labels_on_image_array(
                im,
                boxes,
                classes.astype(np.int32),
                scores,
                self._labels_dict,
                use_normalized_coordinates=True,
                line_thickness=1,
                min_score_thresh=self._retrieval_thresh)
        cv2.imshow('Enhanced image with labels', cv2.resize(im, (im.shape[1] * 2, im.shape[0] * 2)))
        cv2.waitKey(0)

    def _retrival_threshold_cutoff(self, boxes, scores, classes):
        idx_to_keep = np.where(np.greater_equal(np.squeeze(scores), self._retrieval_thresh))
        boxes = np.squeeze(boxes)[idx_to_keep]
        classes = np.squeeze(classes.astype(int))[idx_to_keep]
        scores = np.squeeze(scores)[idx_to_keep]
        return boxes, scores, classes

    def _load_labels(self, labels):
        """
        :type labels: str expected json file
        :return:
        """
        path_to_labels = os.path.join(self._path_to_root + LABELS_DIR, labels)
        with open(path_to_labels, "r") as f:
            raw_dict = json.load(f)
        # reformatting with key as int
        return {int(k): v for k, v in raw_dict.items()}


def _load_images(path_to_images):
    filenames = [os.path.join(path_to_images, f) for f in os.listdir(path_to_images) if
                 f.endswith(".png") and 'gaus' in f]
    n_img = min(100, len(filenames))
    filenames = sample(population=filenames, k=n_img)
    return [Image.open(im_name, mode='r') for im_name in filenames]


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    path = "/shared_experiments/training/train/ckpt/train_events_0/exported" \
           "/frozen_inference_graph.pb"
    images = _load_images("/shared_experiments/testing/extracted_frames")
    retrieval_thresh = .3
    net = EventBasedObjDet(path, retrieval_thresh=retrieval_thresh)
    for im in images:
        im = np.array(im)
        net.run_inference_on_img_and_plot(im)
