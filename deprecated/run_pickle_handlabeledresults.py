import os
import sys
from pprint import pprint

import cv2
import numpy as np
import pandas as pd

sys.path.append(os.getcwd()[:os.getcwd().index('objdetection')])
from utils_visualisation import fun_general
from objdetection.deprecated.handlabeler import Labeler
import json
import pickle


def dump_example(file, format="tfrec"):
    if format == "tfrec":
        name = file["frame_name"][0:-4] + ".tfrec"
        full_path = os.path.join(output_dir, name)
        with open(full_path, 'wb') as f:
            pickle.dump(file, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        name = file["frame_name"][0:-4] + ".pickle"
        full_path = os.path.join(output_dir, name)
        with open(full_path, 'wb') as f:
            pickle.dump(file, f, protocol=pickle.HIGHEST_PROTOCOL)


def remove_fields(detected_obj, keys_to_remove=("height", "width")):
    """	Parse the bounding boxes detection
    :param detected_obj: dictionary
    :param keys_to_remove:
    :return:
    """
    for key in keys_to_remove:
        try:
            detected_obj.pop(key)
        except KeyError:
            print("Key does not exist!")
        except AttributeError:
            return []
    return detected_obj


def loadframe(frame_name):
    frame_full_path = os.path.join(frames_dir, frame_name)
    try:
        im = cv2.imread(frame_full_path)
    except IOError:
        print("Error reading image", frame_full_path)
        im = None
    return im


def format_fields(labels, detected_objs):
    new = {"boxes":  [],
           "labels": [],
           "ids":    []
           }
    if detected_objs[0]:
        for obj in detected_objs:
            box = np.array([obj["p1"]["y"], obj["p1"]["x"], obj["p2"]["y"],
                            obj["p2"]["x"]])
            new["boxes"].append(box)
            new["labels"].append(obj["label"])
            label_id = [labels[key]["id"] for key in labels.keys() if
                        labels[key]["name"] == obj["label"]]
            assert len(label_id) == 1
            new["ids"].append(label_id[0])
        new["boxes"] = np.vstack(new["boxes"])
        new["ids"] = np.stack(new["ids"])
        return new
    else:
        for key in new.keys():
            new[key] = np.array(new[key])
        return new


def main(data):
    """ Receive as input the dictionary of elements parsed.
    Formats the fields of interest as a dictionary and dumps them into pickle.
    :param data:
    :return:
    """

    with open(path_to_labels, "r") as f:
        raw_dict = json.load(f)
        # reformatting with key as int
        labels_dict = {int(k): v for k, v in raw_dict.items()}
    handLabeler = Labeler(labels_dict)
    for row_idx, row in enumerate(data["status"]):
        if resume_number < row_idx:
            detections_parsed = json.loads(row)
            detections_cleaned = list(map(remove_fields, detections_parsed))
            detections_formatted = format_fields(labels_dict,
                                                 detections_cleaned)
            detections_formatted.update(
                    {"frame_name": data["original_filename"][row_idx]})
            detections_formatted.update(
                    {'frame': loadframe(detections_formatted["frame_name"])})
            fun_general.visualize_boxes_and_labels_on_image_array(
                    detections_formatted["frame"],
                    detections_formatted["boxes"],
                    detections_formatted["ids"],
                    None,
                    labels_dict,
                    use_normalized_coordinates=True,
                    line_thickness=1
            )
            print("Processing of ", detections_formatted["frame_name"])
            detections_formatted["boxes"], detections_formatted["ids"] = \
                handLabeler.interface_click_callback(
                        detections_formatted["frame"],
                        detections_formatted["boxes"],
                        detections_formatted["ids"])
            pprint(detections_formatted["ids"])
            pprint(detections_formatted["boxes"])
            dump_example(detections_formatted)
            print("Dumped %d of %d" % (row_idx, len(data["status"])))


if __name__ == '__main__':
    path_to_target = "/home/ale/datasets/zuriscapes/hand_labeling/291f7a3a" \
                     "-229c-472a-9d23-c7ae70d79bf0.csv"
    output_dir = "/home/ale/datasets/zuriscapes/hand_labeling/pickle/"
    frames_dir = "/home/ale/datasets/zuriscapes/hand_labeling" \
                 "/images_hand_labeling"
    path_to_labels = "/home/ale/git_cloned/DynamicVisionTracking" \
                     "/objdetection/" \
                     "SSDneuromorphic/labels/zauronscapes_label_map.json"
    resume_number = 1554
    if not os.path.exists(path_to_target):
        raise ValueError("Select an existing path!")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    else:
        print(
                "The output folder already exists, make sure you do not "
                "overwrite "
                "anything important!")

    data_pd = pd.read_csv(path_to_target)
    values = data_pd.get_values()
    headers = list(data_pd)
    data_dict = dict(
            zip(headers, np.squeeze(np.hsplit(values, values.shape[1]))))
    main(data_dict)
    print("Execution terminated!")
