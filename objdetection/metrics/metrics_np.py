"""
author: az

Numpy implementation to compute performance metrics in object detection.
Metrics from the Pascal VOC challenge
https://link.springer.com/article/10.1007%2Fs11263-014-0733-5
"""
import numpy as np


def gather_stats_on_single_batch(pr_labels, pr_scores, pr_bboxes, gt_labels, gt_bboxes, num_classes,
                                 cut_off_conf, iou_match_thresh=0.5, ):
    """
    :param pr_labels: predictions labels shape (P,)
    :param pr_scores: predictions confidence score shape (P,)
    :param pr_bboxes: predictions bboxes in normalized coord. shape (P, 4)
    :param gt_labels: ground-truth labels shape (G,)
    :param gt_bboxes: ground-truth bboxes in normalized coord. shape (G, 4)
    :param num_classes: number of classes/labels
    :param cut_off_conf: cut off confidence threshold to be considered a prediction (pr_scores)
    :param iou_match_thresh: iou minimum overlap threshold to consider a possible match
    :return n_gt_d, tp_d, fp_d: dictionaries keyed by class/label with n_gt=number of gt bboxes,
    tp=true-positive, fp=false-positive
    """
    n_gt_d, tp_d, fp_d = {}, {}, {}
    # clean prediction below threshold
    over_conf_thresh = np.greater_equal(pr_scores, cut_off_conf)
    pr_labels = pr_labels[over_conf_thresh]
    pr_scores = pr_scores[over_conf_thresh]
    pr_bboxes = pr_bboxes[over_conf_thresh]

    for c in range(1, num_classes + 1):
        # number of gt for category
        n_gt_d[c] = np.count_nonzero(np.equal(gt_labels, c))
        tp_d[c] = 0
        fp_d[c] = 0
        # mask classes
        class_mask = np.equal(pr_labels, c)
        plabels_masked = pr_labels[class_mask]
        pscores_masked = pr_scores[class_mask]
        pbboxes_masked = pr_bboxes[class_mask]
        gt_matched = np.zeros_like(gt_labels, dtype=np.bool)
        for pred_idx, pred in enumerate(plabels_masked):
            if len(gt_bboxes) != 0:
                jaccard = bboxes_jaccard(pbboxes_masked[pred_idx], gt_bboxes)
                # filter out overlapping with other classes
                jaccard = np.multiply(jaccard, np.equal(gt_labels, c))
                if len(jaccard) != 0:
                    # Best fit, checking it's above thresholds.
                    idx_max = np.argmax(jaccard, axis=0)
                    jaccard_max = jaccard[idx_max]
                    overlap_bool = np.greater_equal(jaccard_max, iou_match_thresh)
                    existing_match = gt_matched[idx_max]

                    # TP: match & no previous match and FP: previous match | no match.
                    tp_bool = np.logical_and(overlap_bool, np.logical_not(existing_match))
                    tp_d[c] = tp_d[c] + 1 if tp_bool else tp_d[c]
                    # FP: redundant check (could we simply use not tp_bool?!)
                    fp_bool = np.logical_or(existing_match, np.logical_not(overlap_bool))
                    fp_d[c] = fp_d[c] + 1 if fp_bool else fp_d[c]

                    # Update ground-truth match.
                    gt_matched[idx_max] = True if tp_bool else False
                else:
                    fp_d[c] += 1
            else:
                fp_d[c] += 1

    return n_gt_d, tp_d, fp_d


def bboxes_jaccard(box_ref, bboxes):
    """ Compute jaccard score between a reference box and a collection
    of bounding boxes.
    :param box_ref: (4,) Tensor with reference bounding box.
    :param bboxes: (N, 4) Tensor, collection of bounding boxes.
    :return (N,) Tensor with Jaccard scores.
    """
    # Intersection bbox and volume.
    int_ymin = np.maximum(bboxes[:, 0], box_ref[0])
    int_xmin = np.maximum(bboxes[:, 1], box_ref[1])
    int_ymax = np.minimum(bboxes[:, 2], box_ref[2])
    int_xmax = np.minimum(bboxes[:, 3], box_ref[3])
    h = np.maximum(int_ymax - int_ymin, 0.)
    w = np.maximum(int_xmax - int_xmin, 0.)
    # Volumes.
    inter_vol = h * w
    union_vol = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1]) + \
                (box_ref[2] - box_ref[0]) * (box_ref[3] - box_ref[1]) - inter_vol
    jaccard = np.divide(inter_vol, union_vol,
                        out=np.zeros_like(inter_vol), where=union_vol != 0)
    return jaccard
