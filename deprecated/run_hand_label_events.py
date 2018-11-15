import json
import os
import pickle
from pprint import pprint

import numpy as np

from objdetection.deprecated import input_formatter_deprecated, load_recording_deprecated
from objdetection.deprecated.handlabeler import Labeler

GLOBAL_COUNT = 0


def _load_dict():
    with open(PATH_TO_LABELS, "r") as f:
        raw_dict = json.load(f)
        # reformatting with key as int
        labels_dict = {int(k): v for k, v in raw_dict.items()}
    return labels_dict


def _dump_example(file):
    name = file["frame_name"][0:-4] + ".pickle"
    full_path = os.path.join(OUTPUT_DIR, name)
    try:
        with open(full_path, 'wb') as f:
            pickle.dump(file, f, protocol=pickle.HIGHEST_PROTOCOL)
    except IOError:
        print("io error?!")
    return


def main():
    """ Receive as input the dictionary of elements parsed.
    Formats the fields of interest as a dictionary and dumps them into pickle.
    :return:
    """
    # initialize instruments
    global GLOBAL_COUNT
    loader = load_recording_deprecated.Loader
    formatter = input_formatter_deprecated.SAE(EVENTS_FORMAT,
                                               time_span=EVENTS_FORMAT_TIMESPAN)
    labels_dict = _load_dict()
    handLabeler = Labeler(labels_dict)
    # load recordings names and frames(names only) as done in 
    # pickle_handlabeldresults
    recordings = [os.path.join(RECORDINGS_PATH, r) for r in
                  os.listdir(RECORDINGS_PATH)]
    recordings = sorted(recordings)
    for rec_idx, rec_fold in enumerate(recordings):
        events_dict = loader.load_events(fold=rec_fold)
        frames_list = loader.load_frames(fold=rec_fold)
        frames_list = sorted(frames_list)
        _, rec_name = os.path.split(rec_fold)
        for s_idx in range(len(frames_list)):
            if RESUME_NUMBER < GLOBAL_COUNT:
                _, frame_name = os.path.split(frames_list[s_idx][0])
                name_to_match = rec_name + "_" + frame_name
                detected = {"frame_name": name_to_match}
                prev_ts = max(0, frames_list[s_idx][1] - formatter.time_span)
                events = formatter.crop_and_format_events(
                        events_dict, frame_ts=frames_list[s_idx][1],
                        previous_ts=prev_ts)
                events = (events * 255 / 2 + 255 / 2).astype(np.uint8)
                events = np.stack([events] * 3, axis=-1)
                detected.update({"frame": events})
                detected.update({"boxes": np.array([], dtype=np.float32)})
                detected.update({"ids": np.array([], dtype=np.int64)})
                print("Processing of ", detected["frame_name"])

                detected["boxes"], detected["ids"] = \
                    handLabeler.interface_click_callback(
                            detected["frame"],
                            detected["boxes"],
                            detected["ids"])
                _dump_example(detected)
                pprint(detected["ids"])
                pprint(detected["boxes"])
                print("Dumped %d" % GLOBAL_COUNT)
            GLOBAL_COUNT += 1


if __name__ == '__main__':
    # RECORDINGS_PATH = "/home/ale/encoder/zuriscapes/hand_labeling_night" \
    #                   "/uzhNIGHT/"
    RECORDINGS_PATH = "/media/andya/EXTSSD_T3/crossmodal_mt/rosbags/official/hand_labelling"
    # OUTPUT_DIR = "/home/ale/encoder/zuriscapes/hand_labeling_night/pickle/"
    OUTPUT_DIR = "/media/andya/EXTSSD_T3/crossmodal_mt/rosbags/official/ODN_NE_E0_2018-05-01-21" \
                 "-27-26/"
    # PATH_TO_LABELS = "/home/ale/git_cloned/CrossModalTL/objdetection/" \
    #                  "SSDneuromorphic/labels/zauron_label_map.json"
    PATH_TO_LABELS = "/home/andya/crossmodal_mt/catkin_ws/src/zauron/resources/labels" \
                     "/zauron_label_map.json"
    RESUME_NUMBER = 100
    EVENTS_FORMAT = "_11gaus"
    EVENTS_FORMAT_TIMESPAN = 40 / 1000

    if not os.path.exists(RECORDINGS_PATH):
        raise ValueError("Select an existing path!")
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    else:
        print(
                "The output folder already exists, make sure you do not "
                "overwrite anything important!")
    main()
    print("Execution terminated!")
