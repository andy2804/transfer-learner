import os
from shutil import copy2


def main(recordings_dir, target_dir):
    records_names = os.listdir(recordings_dir)
    for rec_name in records_names:
        rec_path = os.path.join(recordings_dir, rec_name)
        im_names = os.listdir(os.path.join(rec_path, "images/"))
        for im in im_names:
            src = os.path.join(os.path.join(rec_path, "images/"), im)
            dst = os.path.join(target_dir, rec_name + "_" + im)
            try:
                copy2(src, dst=dst)
            except IOError:
                print("error occured during copy")


if __name__ == '__main__':
    recordings_dir = "/media/ale/ZILLY_DISK/ALESSANDRO/zuriscapes_official" \
                     "/hand_labeling_night/uzh"
    target_dir = "/home/ale/datasets/zuriscapes/hand_labeling_night/"
    assert os.path.exists(recordings_dir)
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    main(recordings_dir=recordings_dir, target_dir=target_dir)
