"""
author: az
"""
import unittest

from objdetection.evaluator.evaluator import EvaluatorFrozenGraph


class EvaluatorFrozenGraphTest(unittest.TestCase):
    num_classes = 5
    thresh_levels = 10
    raw_stats = {
        # todo
    }
    acc_rec = {
        # todo
    }

    # def test_compute_acc_rec(self):
    #    acc_rec = EvaluatorFrozenGraph.compute_acc_rec(self.raw_stats, self.num_classes)
    #    self.assertDictEqual(acc_rec, self.acc_rec)

    def test_wilson_ci(self):
        ns = [10, 200]
        n = [500, 400]
        ci = [0.95] * len(n)
        mean_gt = []
        interval_gt = []
        for i in range(len(n)):
            with self.subTest("wilson {:d} subtest".format(i)):
                mean, interval = EvaluatorFrozenGraph.wilson_ci(ns[i], n[i], ci[i])
                print("mean: {:.2f}".format(mean))
                print("mean: {:.4f}".format(interval))
