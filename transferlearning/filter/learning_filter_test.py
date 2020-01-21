"""
author: aa
"""
import unittest


class LearningFilterTest(unittest.TestCase):
    ##########################################
    # parameters used by the class for tests #
    ##########################################
    min_object_size = 80
    score_thresh = 20

    ################
    # actual tests #
    ################

    def test_init(self):
        # TODO implement
        pass

        # with self.assertRaises(ContractNotRespected):
        #     ber = Bernoulli(1, -1)

    def test_rgb_mode(self):
        # TODO implement
        pass

        # lb_gt = [0, .2591, .5921]
        # ub_gt = [.3445, .4526, .6078]
        # for i in range(len(self.n)):
        #     ber = Bernoulli(self.ns[i], self.n[i])
        #     with self.subTest("Wilson {:d} subtest [computation]".format(i)):
        #         lb, ub = ber.get_confidence_interval(self.confidence[i], "wilson")
        #         print(lb, "<->", ub)
        #         # self.assertAlmostEqual(lb, lb_gt[i], places=4)
        #         # self.assertAlmostEqual(ub, ub_gt[i], places=4)

    def test_events_mode(self):
        # TODO implement
        pass

        # for i in range(len(self.n)):
        #     ber = Bernoulli(self.ns[i], self.n[i])
        #     with self.subTest("Clopper pearson {:d} subtest [computation]".format(i)):
        #         lb, ub = ber.get_confidence_interval(self.confidence[i], "clopper_pearson")
        #         print(lb, "<->", ub)
        #         # self.assertAlmostEqual(lb, lb_gt[i], places=4)
        #         #self.assertAlmostEqual(ub, ub_gt[i], places=4)

