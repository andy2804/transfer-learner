"""
author: az
"""
import tensorflow as tf


# fixme move these function in a general SSD class
# todo update names also with ssd_common

# =================== Encode/decode groundtruth
def match_gt_to_db(xcent_gt, ycent_gt, w_gt, h_gt, labels_gt, dboxes_lookup_dict,
                   iou_thresh, prior_scaling):
    """Function that matches the ground truth boxes to the default boxes.
    Loops over the feature maps calling _encode_layer function
    """
    y_true_label = []
    y_true_iou_score = []
    y_true_loc = []
    # for loop over the feature maps
    for lay_name, dboxes in sorted(dboxes_lookup_dict.items()):
        with tf.name_scope("match_gt2db_" + lay_name):
            true_label, true_iou_scores, true_loc = \
                _encode_layer(xcent_gt, ycent_gt, w_gt, h_gt, labels_gt, dboxes,
                              iou_thresh, prior_scaling)
            # record encoding for that layer
            y_true_label.append(true_label)
            y_true_iou_score.append(true_iou_scores)
            y_true_loc.append(true_loc)
    # final reshaping of all layers
    y_true_label = tf.concat(y_true_label, axis=0)
    y_true_iou_score = tf.concat(y_true_iou_score, axis=0)
    y_true_loc = tf.concat(y_true_loc, axis=0)
    y_true_loc = tf.reshape(y_true_loc, shape=[-1])  # fixme is it needed?
    return y_true_label, y_true_iou_score, y_true_loc


def _encode_layer(xcent_gt, ycent_gt, w_gt, h_gt, labels_gt, dboxes, iou_thresh,
                  prior_scaling, dtype=tf.float32):
    """ Given a layer(i.e a feature map) matches the gt boxes to the default
    boxes.
    :param xcent_gt: coordinates of ground-truth boxes
    :param ycent_gt: coordinates of ground-truth boxes
    :param w_gt: coordinates of ground-truth boxes
    :param h_gt: coordinates of ground-truth boxes
    :param labels_gt: labels for ground-truth boxes
    :param dboxes:
    :param iou_thresh: overlapping threshold to match
    :param prior_scaling: scaling of the encoding of the localization offset
    :param dtype:
    :return: encoded labels, scores(overlapping of gt to db) and localization
    """
    # Default boxes coordinates and volume.
    xref, yref, wref, href = tf.unstack(dboxes, axis=1)
    vol_dbboxes = wref * href

    # Initialize tensors...
    shape = (xref.shape[0])
    encoded_label = tf.zeros(shape, dtype=tf.int64)
    encoded_iou = tf.zeros(shape, dtype=dtype)
    encoded_loc = tf.zeros((xref.shape[0], 4), dtype=dtype)

    def jaccard_overlap(x_gt, y_gt, w_gt, h_gt):
        """ Calculate intersection, i.e. area of overlap among the gt_box and
        all the default boxes of this feature map layer (could be 0)
        """
        x_overlap = tf.maximum(0.0,
                               tf.minimum(xref + wref / 2., x_gt + w_gt / 2.) -
                               tf.maximum(xref - wref / 2., x_gt - w_gt / 2.))
        y_overlap = tf.maximum(0.0,
                               tf.minimum(yref + href / 2., y_gt + h_gt / 2.) -
                               tf.maximum(yref - href / 2., y_gt - h_gt / 2.))
        intersection = tf.multiply(x_overlap, y_overlap)
        # Calculate union
        union = tf.multiply(w_gt, h_gt) + vol_dbboxes - intersection
        jaccard = tf.divide(intersection, union)
        return jaccard

    def condition(idx, foo1, foo2, foo3):
        """
        :return: true if there are still gt_boxes to be matched
        """
        return tf.less(idx, tf.shape(labels_gt)[0])

    def body(idx, enc_labels, enc_iou, enc_loc):
        """
        :param enc_labels: gt matched category
        :param enc_iou: iou scores (gt and dboxes)
        :param enc_loc: encoded offset with respect to dboxes
        """
        gt_label = labels_gt[idx]
        iou = jaccard_overlap(
                xcent_gt[idx], ycent_gt[idx], w_gt[idx], h_gt[idx])
        # Update only where we have overlapping and we beat previous iou scores
        update_mask = tf.logical_and(
                tf.greater_equal(iou, iou_thresh), tf.greater(iou, enc_iou))
        imask = tf.cast(update_mask, tf.int64)
        enc_labels = tf.multiply(imask, gt_label) + (1 - imask) * enc_labels
        enc_iou = tf.where(update_mask, iou, enc_iou)
        x_loc = (xcent_gt[idx] - xref) / wref / prior_scaling[0]
        y_loc = (ycent_gt[idx] - yref) / href / prior_scaling[1]
        w_loc = tf.log(w_gt[idx] / wref) / prior_scaling[2]
        h_loc = tf.log(h_gt[idx] / href) / prior_scaling[3]
        offset = tf.stack([x_loc, y_loc, w_loc, h_loc], axis=-1)
        enc_loc = tf.where(update_mask, offset, enc_loc)
        return [idx + 1, enc_labels, enc_iou, enc_loc]

    # while loop over the gt objects
    i = 0
    [i, encoded_label, encoded_iou, encoded_loc] = \
        tf.while_loop(condition, body,
                      [i, encoded_label, encoded_iou, encoded_loc])

    return encoded_label, encoded_iou, encoded_loc


def decode_db(pred_loc_offset, dboxes_lookup_tensor, prior_scaling):
    """ Receives as input the predicted offset for localization of each dbox
    :param prior_scaling:
    :param pred_loc_offset: output of the network
    :param dboxes_lookup_tensor: tensor with all the default boxes coordinates
    :return x_center, y_center, width, height of the predicted boxes in frame
    coordinates
    """
    xref, yref, wref, href = tf.unstack(dboxes_lookup_tensor, axis=1)
    xpred, ypred, wpred, hpred = tf.unstack(pred_loc_offset, axis=2)

    # compute actual values
    xreal = (xpred * wref * prior_scaling[0]) + xref
    yreal = (ypred * href * prior_scaling[1]) + yref
    wreal = tf.exp(wpred * prior_scaling[2]) * wref
    hreal = tf.exp(hpred * prior_scaling[3]) * href
    return xreal, yreal, wreal, hreal


def bboxes_center_format_to_minmax(xcent, ycent, width, height):
    """ Function than handles box coordinates transformation.
    All values are expected between 0 and 1
    """
    xmin = tf.maximum(0., xcent - (width / 2.))
    ymin = tf.maximum(0., ycent - (height / 2.))
    xmax = tf.minimum(1., xcent + (width / 2.))
    ymax = tf.minimum(1., ycent + (height / 2.))
    return xmin, ymin, xmax, ymax


def bboxes_minmax_format_to_center(xmin, ymin, xmax, ymax):
    """ Function than handles box coordinates transformation.
    All values are expected between 0 and 1
    """
    width = xmax - xmin
    height = ymax - ymin
    xcent = xmin + width / 2
    ycent = ymin + height / 2
    return xcent, ycent, width, height


def format_gt(xcent, ycent, w, h, gt_labels, gt_size):
    """ Formats gt_boxes to minmax format and zero pads
    :param gt_labels: ground-truth labels
    :param gt_size: ground-truth size for zero-padding (necessary for batch
    stacking)
    :return: tensor of gt boxes padded with zeros (s.t. they can be stacked
    in batches) in format [ymin, xmin, ymax, xmax]
    """
    target_len = tf.constant([gt_size], dtype=tf.int32)
    xmin, ymin, xmax, ymax = bboxes_center_format_to_minmax(xcent, ycent, w, h)
    gt_boxes = tf.stack([ymin, xmin, ymax, xmax], axis=1)
    diff_len = target_len - tf.shape(gt_boxes)[0]
    pad = tf.zeros(
            tf.concat([diff_len, tf.constant([4], dtype=tf.int32)], axis=0), dtype=tf.float32)
    gt_boxes = tf.concat([gt_boxes, pad], axis=0)
    gt_labels = tf.concat([gt_labels, tf.zeros(diff_len, dtype=tf.int64)], axis=0)
    return gt_labels, gt_boxes
