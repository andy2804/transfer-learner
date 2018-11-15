"""
author: az

Tensorflow implementation to compute performance metrics in object detection
"""
import tensorflow as tf


def gather_stats_on_single(c_filter, pr_labels, pr_scores, pr_bboxes,
                           gt_labels, gt_bboxes, jaccard_thresh=0.5,
                           conf_thresh=0.5, scope=None):
    """
    Matching a collection of detected boxes with ground-truth values.
    Does not accept batched-inputs.
    The algorithm goes as follows: for every detected box, check
    if one groundtruth box is matching(label+overlap). If none, then it is
    considered as False Positive.
    If the groundtruth box is already matched with another one, it also counts
    as a False Positive (predictions are sorted by score).
    We consider as False Negative if a gt box has not an overlapping prediction
    (more than matching_threshold), independently from the labels.    
    All arguments are likely to be zero padded, hence zero-class objects are
    ignored.
    Args:
        :param c_filter: class filter id.
        :param pr_labels:(N,) predicted classes.
        :param pr_scores: (N,) predicted scores.
        :param pr_bboxes: (N, 4) predicted boxes.
        :param gt_labels: (M,) Groundtruth labels.
        :param gt_bboxes: (M, 4) Groundtruth bounding boxes.
        :param jaccard_thresh: Threshold for a sufficient iou match.
        :param conf_thresh: minimum confidence level to be considered a
        prediction.
        :return _:
            n_gbboxes: Scalar Tensor with # of groundtruth boxes.
            tp_match: Scalar Tensor containing with # of True Positives.
            fp_match: Scalar Tensor containing with # ofFalse Positives.
    """

    with tf.name_scope(scope, 'bboxes_match_single'):
        # This can be computed for free
        n_gtboxes = tf.count_nonzero(tf.equal(gt_labels, c_filter))
        # filter out predictions of other classes
        c_filter = tf.cast(c_filter, dtype=pr_labels.dtype)
        class_filter = tf.multiply(c_filter, tf.ones_like(pr_labels))
        class_filter_bool = tf.equal(pr_labels, class_filter)
        class_filter_keep = tf.where(class_filter_bool)
        pr_labels = tf.gather(pr_labels, class_filter_keep)
        pr_scores = tf.gather(pr_scores, class_filter_keep)
        pr_bboxes = tf.gather(pr_bboxes, class_filter_keep)
        p_size = tf.shape(pr_labels)[0]
        # initialize variables
        tp_match = tf.zeros(shape=[1], dtype=tf.int64)
        fp_match = tf.zeros(shape=[1], dtype=tf.int64)
        gt_match = tf.cast(tf.zeros_like(gt_labels), dtype=tf.bool)

        # Loop over returned objects.
        def m_condition(i, tp, fp, gt_taken):
            return tf.less(i, p_size)

        def m_body(i, tp, fp, gt_taken):
            # Jaccard score with groundtruth bboxes.
            p_box = tf.squeeze(pr_bboxes[i])
            p_score = tf.squeeze(pr_scores[i])
            jaccard = bboxes_jaccard(p_box, gt_bboxes)
            # filter out overlapping with other classes
            jaccard = tf.multiply(jaccard, tf.cast(tf.equal(gt_labels, c_filter),
                                                   dtype=jaccard.dtype))

            # Best fit, checking it's above thresholds.
            idx_max = tf.cast(tf.argmax(jaccard, axis=0), tf.int64)
            jaccard_max = jaccard[idx_max]
            overlap_bool = tf.greater_equal(jaccard_max, jaccard_thresh)
            existing_match = gt_taken[idx_max]
            above_conf_bool = tf.greater_equal(p_score, conf_thresh)

            # TP: match & no previous match and FP: previous match | no match.
            tp_bool = tf.logical_and(
                    tf.logical_and(overlap_bool,
                                   tf.logical_not(existing_match)),
                    above_conf_bool)
            tp = tf.cond(tp_bool, lambda: tf.add(tp, 1), lambda: tp)
            # FP: redundant check (could we simply use not tp_bool?!)
            fp_bool = tf.logical_and(
                    tf.logical_or(existing_match, tf.logical_not(overlap_bool)),
                    above_conf_bool)
            fp = tf.cond(fp_bool, lambda: tf.add(fp, 1), lambda: fp)

            # Update ground-truth match.
            range_mask = tf.cast(tf.range(tf.shape(gt_taken)[0]),
                                 dtype=tf.int64)
            idx_update_mask = tf.equal(range_mask, idx_max)
            if_true_update = tf.logical_and(tp_bool,
                                            tf.cast(tf.ones_like(gt_taken),
                                                    dtype=tf.bool))
            gt_taken = tf.where(idx_update_mask, if_true_update, gt_taken)
            return [i + 1, tp, fp, gt_taken]

        # Main loop definition.
        i = 0
        [i, tp_match, fp_match, gt_match] = tf.while_loop(m_condition, m_body,
                                                          [i, tp_match,
                                                           fp_match, gt_match],
                                                          parallel_iterations=10,
                                                          back_prop=False)
        return n_gtboxes, tp_match, fp_match


def gather_stats_on_batch(plabels, pscores, pbboxes,
                          glabels, gbboxes, num_classes, cut_off_conf,
                          iou_match_thresh=0.5, scope=None):
    """
    Given a batch of predictions and the corresponding ground truth,
    compute for each class the number of TruePositive, FalsePositive and
    Number of gt.
    :param plabels: (b, N,) predictions
    :param pscores: (b, N,) predictions
    :param pbboxes: (b, N, 4) predictions
    :param glabels: (b, M,) ground-truth
    :param gbboxes: (b, M, 4) ground-truth
    :param num_classes: number of possible classes ()
    :param iou_match_thresh: necessary minimum overlap
    :param cut_off_conf: minimum retrival confidence threshold
    :param scope:
    :return: dictionaries with keys as class
    """
    n_gt_d, tp_d, fp_d = {}, {}, {}
    # loop across possible classes, class 0 (background is neglected)
    for c in range(1, num_classes):
        with tf.name_scope(scope, 'bboxes_match_batch_' + str(c)):
            # map fun across the batch
            n_gt, tp, fp = tf.map_fn(
                    lambda x: gather_stats_on_single(c, x[0], x[1], x[2], x[3],
                                                     x[4],
                                                     iou_match_thresh,
                                                     cut_off_conf),
                    (plabels, pscores, pbboxes, glabels, gbboxes),
                    dtype=(tf.int64, tf.int64, tf.int64),
                    back_prop=False, parallel_iterations=10, swap_memory=True)
            n_gt_d[c] = tf.reduce_sum(n_gt)
            tp_d[c] = tf.reduce_sum(tp)
            fp_d[c] = tf.reduce_sum(fp)
    return n_gt_d, tp_d, fp_d


def bboxes_jaccard(box_ref, bboxes, name=None):
    """
    Compute jaccard score between a reference box and a collection
    of bounding boxes.
    :param box_ref: (4,) Tensor with reference bounding box.
    :param bboxes: (N, 4) Tensor, collection of bounding boxes.
    :return (N,) Tensor with Jaccard scores.
    """
    with tf.name_scope(name, 'bboxes_jaccard'):
        # Intersection bbox and volume.
        int_ymin = tf.maximum(bboxes[:, 0], box_ref[0])
        int_xmin = tf.maximum(bboxes[:, 1], box_ref[1])
        int_ymax = tf.minimum(bboxes[:, 2], box_ref[2])
        int_xmax = tf.minimum(bboxes[:, 3], box_ref[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        # Volumes.
        inter_vol = h * w
        union_vol = (bboxes[:, 2] - bboxes[:, 0]) * (
                bboxes[:, 3] - bboxes[:, 1]) + \
                    (box_ref[2] - box_ref[0]) * (
                            box_ref[3] - box_ref[1]) - inter_vol
        jaccard = safe_divide(inter_vol, union_vol, 'jaccard')
        return jaccard


def safe_divide(numerator, denominator, name):
    """Divides two values, returning 0 if the denominator is <= 0.
        Args:
          numerator: A real `Tensor`.
          denominator: A real `Tensor`, with dtype matching `numerator`.
          name: Name for the returned op.
        Returns:
          0 if `denominator` <= 0, else `numerator` / `denominator`
        """
    return tf.where(tf.greater(denominator, 0),
                    tf.divide(numerator, denominator),
                    tf.zeros_like(numerator),
                    name=name)


def compute_acc_rec(n_gt, tp, fp):
    # todo documentation, maybe safety checks?
    acc, rec = {}, {}
    for c in n_gt.keys():
        acc[c] = safe_divide(tf.cast(tp[c], dtype=tf.float64),
                             tf.cast(tf.add(tp[c], fp[c]), dtype=tf.float64),
                             name=None)
        rec[c] = safe_divide(tf.cast(tp[c], dtype=tf.float64),
                             tf.cast(n_gt[c], dtype=tf.float64), name=None)
    return acc, rec


def compute_acc_rec_np(n_gt, tp, fp):
    # todo documentation, maybe safety checks?
    acc, rec = {}, {}
    for c in n_gt.keys():
        acc[c] = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0
        rec[c] = tp[c] / n_gt[c] if n_gt[c] > 0 else 0
    return acc, rec
