from pprint import pprint


class StatsLogger:
    def __init__(self, labels):
        self.stats = {}
        for key in labels.keys():
            self.stats[key] = 0

    def update_stats(self, obj_categories):
        for lab in obj_categories:
            self.stats[lab] += 1

    def print(self, ):
        pprint(self.stats)
