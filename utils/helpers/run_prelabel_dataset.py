import os
import sys

import numpy as np
from PIL import Image

PROJECT_ROOT = os.getcwd()[:os.getcwd().index('utils')]
sys.path.append(PROJECT_ROOT)
PROJECT_ROOT = os.getcwd()[:os.getcwd().index('WormholeLearning')]
sys.path.append(PROJECT_ROOT)

from utils.static_helper import load_labels
from objdetection.detector.detector import Detector


def _read_rgb_filenames(dir):
    assert os.path.isdir(dir)
    return [os.path.join(dir, f) for f in os.listdir(dir) if
            FILTER_KEY in f and f.endswith(".png")]


def _create_xml_pascal(img_path, img_rgb, classes_remapped, boxes_remapped, label_map):
    base_filename = os.path.splitext(img_path)[0]
    with open(base_filename + '.xml', 'w') as f:
        line = "<annotation>" + '\n'
        f.write(line)
        line = '\t\t<folder>' + "folder" + '</folder>' + '\n'
        f.write(line)
        line = '\t\t<filename>' + base_filename + ".png" + '</filename>' + '\n'
        f.write(line)
        line = '\t\t<source>\n\t\t<database>Kaist</database>\n\t</source>\n'
        f.write(line)
        height, width = img_rgb.shape[:2]
        line = '\t<size>\n\t\t<width>' + str(width) + '</width>\n\t\t<height>' + str(
                height) + '</height>\n\t'
        line += '\t<depth>3</depth>\n\t</size>'
        f.write(line)
        line = '\n\t<segmented>0</segmented>'
        f.write(line)

        for i, label_id in enumerate(classes_remapped):
            ymin, xmin, ymax, xmax = (boxes_remapped[i] * np.array(
                    [height, width, height, width])).astype(np.int)
            line = '\n\t<object>'
            line += '\n\t\t<name>' + label_map[label_id][
                "name"] + '</name>\n\t\t<pose>Unspecified</pose>'
            line += '\n\t\t<truncated>Unspecified</truncated>\n\t\t<difficult>0</difficult>'
            line += '\n\t\t<bndbox>\n\t\t\t<xmin>' + str(xmin) + '</xmin>'
            line += '\n\t\t\t<ymin>' + str(ymin) + '</ymin>'
            line += '\n\t\t\t<xmax>' + str(xmax) + '</xmax>'
            line += '\n\t\t\t<ymax>' + str(ymax) + '</ymax>'
            line += '\n\t\t</bndbox>'
            line += '\n\t</object>\n'
            f.write(line)

        line = "</annotation>" + '\n'
        f.write(line)


def main():
    detector = Detector(net_id=NET_ARCH, labels_net_arch=LABELS_NET, labels_output=LABELS_OUT)
    rgb_files = _read_rgb_filenames(DATASET_DIR)
    rgb_files.sort()
    label_map = load_labels(LABELS_OUT)

    for idx, img_path in enumerate(rgb_files):
        print("\r[ %d / %d ] Processing %s" % (idx, len(rgb_files), os.path.basename(img_path)),
              end='', flush=True)
        img_rgb = np.array(Image.open(img_path))
        # run inference
        obj_detected = detector.run_inference_on_img(img_rgb)
        classes_remapped, scores_remapped, boxes_remapped = detector.remap_labels(
                obj_detected.classes, obj_detected.scores, obj_detected.boxes)

        if len(boxes_remapped) > 0:
            classes_remapped = np.squeeze(classes_remapped, axis=0)
            boxes_remapped = np.squeeze(boxes_remapped, axis=0)
            scores_remapped = np.squeeze(scores_remapped, axis=0)

        _create_xml_pascal(
                img_path, img_rgb, classes_remapped, boxes_remapped, label_map)


if __name__ == '__main__':
    NET_ARCH = 2
    LABELS_NET = "mscoco_label_map.json"
    LABELS_OUT = "zauron_label_map.json"
    DATASET_DIR = "/home/andya/external_ssd/wormhole_learning/dataset/testing_handlabeling" \
                  "/night_sampled_rss_additional"
    FILTER_KEY = 'RGB'
    CUDA_MASK = "0"

    # main
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_MASK
    main()
