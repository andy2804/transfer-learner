"""
author: az

Numpy implementation to compute performance metrics in object detection.
Metrics from the Pascal VOC challenge
https://link.springer.com/article/10.1007%2Fs11263-014-0733-5
"""
import unittest

import numpy as np

from objdetection.meta.performances.metrics_np import gather_stats_on_single_batch


class metricsNpTest(unittest.TestCase):
    """
    coord bboxes = [ymin, xmin, ymax, xmax]
    """
    # inputs
    pr_labels = np.asarray([1, 2, 3, 4, 5, 4, 4, 4])
    pr_scores = np.asarray([.9, .8, .7, .65, .6, .55, .49, .13])
    pr_bboxes = np.asarray([[.5, .12, .73, .45],  # tp
                            [.5, .71, .75, .97],  # tp
                            [.07, .11, .34, .47],  # tp
                            [.48, .8, .8, .89],  # tp
                            [.5, .7, .75, .97],  # fp no 5 in gt
                            [.49, .82, .81, .88],  # fp double detection 4
                            [.07, .1, .34, .4],  # none, low conf
                            [.48, .8, .8, .89]  # none, low conf
                            ])

    # gt
    gt_labels = np.asarray([1, 2, 3, 4])
    gt_bboxes = np.asarray([[.5, .1, .75, .45],  # bottom left
                            [.5, .7, .75, .97],  # bottom right
                            [.067, .1, .34, .45],  # up wide
                            [.48, .8, .8, .89]])  # bottom right (partial overlap with 2)
    num_classes = 5
    cut_off_conf = .5
    # expected output
    classes = list(range(1, num_classes + 1))
    n_gt_d = dict(zip(classes, [1, 1, 1, 1, 0]))
    tp_d = dict(zip(classes, [1, 1, 1, 1, 0]))
    fp_d = dict(zip(classes, [0, 0, 0, 1, 1]))

    def test_parameters_consistency(self):
        self.assertListEqual(list(self.n_gt_d.keys()), list(self.tp_d.keys()))
        self.assertListEqual(list(self.n_gt_d.keys()), list(self.fp_d.keys()))
        self.assertEqual(self.pr_labels.shape[0], self.pr_scores.shape[0])
        self.assertEqual(self.pr_labels.shape[0], self.pr_bboxes.shape[0])
        self.assertEqual(self.pr_bboxes.shape[1], 4)

    def test_iscallable(self):
        self.assertTrue(callable(gather_stats_on_single_batch))

    def test_stats_on_single_batch(self):
        n_gt_dict, tp_dict, fp_dict = gather_stats_on_single_batch(
                self.pr_labels, self.pr_scores, self.pr_bboxes, self.gt_labels,
                self.gt_bboxes, self.num_classes, self.cut_off_conf
        )
        with self.subTest("ground-truth dictionary"):
            self.assertDictEqual(n_gt_dict, self.n_gt_d)
        with self.subTest("true positive dictionary"):
            self.assertDictEqual(tp_dict, self.tp_d)
        with self.subTest("false positive dictionary"):
            self.assertDictEqual(fp_dict, self.fp_d)


if __name__ == '__main__':
    unittest.main()
