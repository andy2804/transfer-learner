import json
import os

root = "/home/ale/Dropbox/Uni/Eth/Thesis/catkin_ws/" \
       "objdetection/SSDneuromorphic/" \
       "datasets/labels/"
file_name2load = "mscoco_label_map.json"
file_name2dump = "zauronscapes_label_map.json"

with open(os.path.join(root, file_name2load), 'r') as f:
    raw_dict = json.load(f)
    # reformatting with key as int
    category_idx = {k: v for k, v in raw_dict.items() if int(k) < 11}

# do more
with open(os.path.join(root, file_name2dump), 'w') as f:
    json.dump(category_idx, f)
print("done!")
