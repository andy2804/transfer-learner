"""
author: aa
"""
import collections

from objdetection.meta.utils_labeler.static_helper import load_labels


class LearningTracker:
    def __init__(self, n_instances=0, labels_file='zauron_label_map.json'):
        self._n_instances = n_instances
        self._n_objects = 0
        self._labels = load_labels(labels_file)
        self._counter = collections.OrderedDict().fromkeys([
            v["name"] for v in self._labels.values()], (0, 0))
        self._instance = collections.OrderedDict()

    def add(self, keep, score, class_id, box, timestamp):
        """
        Instance to statistical information about object detection
        :param class_id:
        :return:
        """
        obj = self._labels[class_id]["name"]
        count = 0
        name = '%d_%s_%02i' % (timestamp, obj, count)
        while name in self._instance:
            count += 1
            name = '%d_%s_%02i' % (timestamp, obj, count)
        self._instance[name] = {
            'learn':     keep,
            'score':     score,
            'object':    obj,
            'bbox':      box,
            'timestamp': timestamp
        }
        self._n_objects += 1
        self._increment_counter(obj, keep)
        return name

    def _increment_counter(self, name, keep):
        accepted = self._counter[name][0]
        total = self._counter[name][1]
        if keep:
            accepted += 1
        total += 1
        self._counter[name] = (accepted, total)

    def change_counter(self, name, change):
        accepted = self._counter[name][0]
        total = self._counter[name][1]
        accepted += change
        self._counter[name] = (accepted, total)

    def _reset_counter(self):
        for obj, values in self._counter.items():
            self._counter[obj] = (0, 0)

    def _update_counter(self):
        self._reset_counter()
        for obj, values in self._instance.items():
            self._increment_counter(values['object'], values['learn'])

    def set_instances(self, max_instances):
        self._n_instances = max_instances

    def get_instances(self):
        return self._n_instances

    def __getitem__(self, key):
        return self._instance[key]

    def __setitem__(self, key, value):
        self._instance[key] = value

    def __contains__(self, key):
        return key in self._instance

    def __copy__(self):
        return self._instance.copy()

    def values(self):
        return self._instance.values()

    def items(self):
        return self._instance.items()

    def pop(self, key=None):
        if key is None:
            return self._instance.popitem(last=True)
        else:
            return self._instance.popitem(key)

    def pretty_print(self, ):
        print('\nLearning Tracker Accepted Objects:\033[K')
        for obj, stats in self._counter.items():
            print('\t%s\t: %d / %d\033[K' % (
                obj.ljust(12), stats[0], stats[1]))
