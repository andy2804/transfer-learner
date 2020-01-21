"""
author: az
"""
import unittest

from contracts import ContractNotRespected

from utils.stats.bernoulli import Bernoulli


class BernoulliTest(unittest.TestCase):
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
    def test_init_exception(self):
        with self.assertRaises(ContractNotRespected):
            ber = Bernoulli(1, -1)

    def test_wilson_ci(self):
        """ground truth from: http://vassarstats.net/prop1.html """

        lb_gt = [0, .2591, .5921]
        ub_gt = [.3445, .4526, .6078]
        for i in range(len(self.n)):
            ber = Bernoulli(self.ns[i], self.n[i])
            with self.subTest("Wilson {:d} subtest [computation]".format(i)):
                lb, ub = ber.get_confidence_interval(self.confidence[i], "wilson")
                print(lb, "<->", ub)
                # self.assertAlmostEqual(lb, lb_gt[i], places=4)
                # self.assertAlmostEqual(ub, ub_gt[i], places=4)

    def test_clopper_pearson_ci(self):
        # todo lb_gt, ub_gt

        for i in range(len(self.n)):
            ber = Bernoulli(self.ns[i], self.n[i])
            with self.subTest("Clopper pearson {:d} subtest [computation]".format(i)):
                lb, ub = ber.get_confidence_interval(self.confidence[i], "clopper_pearson")
                print(lb, "<->", ub)
                # self.assertAlmostEqual(lb, lb_gt[i], places=4)
                #self.assertAlmostEqual(ub, ub_gt[i], places=4)

# def test_compute_acc_rec(self):
#    acc_rec = EvaluatorFrozenGraph.compute_acc_rec(self.raw_stats, self.num_classes)
#    self.assertDictEqual(acc_rec, self.acc_rec)
