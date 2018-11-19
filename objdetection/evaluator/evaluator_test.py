"""
author: az
"""
import unittest

from contracts import ContractNotRespected

from objdetection.evaluator.evaluator import EvaluatorFrozenGraph


class EvaluatorFrozenGraphTest(unittest.TestCase):
    ##########################################
    # parameters used by the class for tests #
    ##########################################
    num_classes = 5
    thresh_levels = 10
    raw_stats = {
        # todo
    }
    acc_rec = {
        # todo
    }

    # Confidence intervals common
    ns = [0, 35, 9000]
    n = [10, 100, 15000]
    confidence = [.95, .95, .95]

    ################
    # actual tests #
    ################
    def test_wilson_ci(self):
        """ground truth from: http://vassarstats.net/prop1.html """

        lb_gt = [0, .2591, .5921]
        ub_gt = [.3445, .4526, .6078]
        for i in range(len(self.n)):
            with self.subTest("Wilson {:d} subtest [computation]".format(i + 1)):
                lb, ub = EvaluatorFrozenGraph.wilson_ci(
                        self.ns[i], self.n[i], self.confidence[i])
                print(lb, "+-", ub)
                self.assertAlmostEqual(lb, lb_gt[i], places=4)
                self.assertAlmostEqual(ub, ub_gt[i], places=4)
        with self.subTest("Wilson [args validity]"):
            with self.assertRaises(ContractNotRespected):
                EvaluatorFrozenGraph.wilson_ci(1, -1, .9)

    def test_clopper_pearson_ci(self):
        # todo
        pass

# def test_compute_acc_rec(self):
#    acc_rec = EvaluatorFrozenGraph.compute_acc_rec(self.raw_stats, self.num_classes)
#    self.assertDictEqual(acc_rec, self.acc_rec)
