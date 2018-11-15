"""
author: az
"""
import unittest

from contracts import ContractNotRespected

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
        ns = [35, 9000]
        n = [100, 15000]
        ci = [.95, .9]
        mean_gt = [.361968, .6]
        interval_gt = [.09191, .0066]
        for i in range(len(n)):
            with self.subTest("Wilson {:d} subtest [computation]".format(i + 1)):
                mean, interval = EvaluatorFrozenGraph.wilson_ci(ns[i], n[i], ci[i])
                self.assertAlmostEqual(mean, mean_gt[i], places=4)
                self.assertAlmostEqual(interval, interval_gt[i], places=4)
        with self.subTest("Wilson [args validity]"):
            with self.assertRaises(ContractNotRespected):
                EvaluatorFrozenGraph.wilson_ci(1, -1)
