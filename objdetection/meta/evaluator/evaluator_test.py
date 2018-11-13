"""
author: az
"""
import unittest

from objdetection.meta.evaluator.evaluator import EvaluatorFrozenGraph


class EvaluatorFrozenGraphTest(unittest.TestCase):
    num_classes = 5
    thresh_levels = 10
    raw_stats = {
        # todo
    }
    acc_rec = {
        # todo
    }

    def test_compute_acc_rec(self):
        acc_rec = EvaluatorFrozenGraph.compute_acc_rec(self.raw_stats, self.num_classes)
        self.assertDictEqual(acc_rec, self.acc_rec)
