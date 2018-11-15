"""
author: az
"""
import numpy as np
import tensorflow as tf

from objdetection.rgb2events.nets import ssd_common
from objdetection.meta.metrics import metrics_tf
from objdetection.meta.datasets import data_augmenter_sae
from objdetection.deprecated.encoder_tfrecord_deprecated import \
    decode_tfrecord
from objdetection.meta.utils_generic.magic_constants import *


# =========================================================================== #
# SSD class definition.
# =========================================================================== #
class SSDNet:
    """ Implementation of the SSD for DAVIS c
    """

    def __init__(self, params=None, options=None, conf_thresh_cutoff=.5):
        """Init the SSD net with some parameters.
        Use the default ones if none provided.
        :param params:
        :param options:
        """
        if isinstance(params, LearnParams):
            self._params = params
        else:
            self._params = DEFAULT_LEARN_PARAMS
        if isinstance(options, ObjDetParams):
            self.opt = options
        else:
            self.opt = DEFAULT_OBJDET_PAR
        self.conf_thresh_cutoff = conf_thresh_cutoff

        # ============= Some useful numbers architecture dependent
        self._fm_cells_tot_n = int(sum([v["size"][0] * v["size"][1]
                                        for v in
                                        self._params.featmap_layers.values()]))
        self._dboxes_tot_n = int(
                sum([v["size"][0] * v["size"][1] * len(v["aspect_ratio"])
                     for v in self._params.featmap_layers.values()]))
        self._predconf_tot_n = self._dboxes_tot_n * self._params.num_classes
        self._predloc_tot_n = self._dboxes_tot_n * 4
        self._dboxes_lookup_dict, self._dboxes_lookup_tensor = \
            self._build_dboxes_table()
        self.augmenter = data_augmenter_sae.DataAugSae(
                flipX_bool=True, flip_polarity_bool=True,
                random_quant_bool=True,
                sample_distorted_bbox_bool=False, sample_time_axis_bool=False,
                random_yshift_bool=True)
        # ============= Placeholders for the model inputs
        self.global_step = tf.Variable(0, trainable=False)
        self.events_in = tf.placeholder(
                tf.float32, shape=self._params.input_shape, name='events_in')
        self.y_db_conf = tf.placeholder(tf.int32, [None, self._dboxes_tot_n],
                                        name='y_db_conf')  # classification 
        # ground-truth labels
        self.y_db_conf_score = tf.placeholder(tf.float32,
                                              [None, self._dboxes_tot_n],
                                              name='y_db_conf')
        self.y_db_loc = tf.placeholder(tf.float32, [None, self._predloc_tot_n],
                                       name='y_db_loc')  # localization 
        # ground-truth labels
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.gt_boxes = tf.placeholder(tf.float32, [None, self.opt.gt_size, 4])
        self.gt_labels = tf.placeholder(tf.int64, [None, self.opt.gt_size])
        # ============= Connect inputs to output(build network and losses)
        self.y_pred_conf, self.y_pred_loc, self.end_points = self._inference(
                weight_decay=self.opt.weight_decay)
        self.loss = self._build_losses()
        self.train_opt = self._build_optimizer()
        self.classes_pred, self.scores_pred, self.boxes_pred = \
            self._decode_predictions()
        self.final_events, self.final_classes, self.final_scores, \
        self.final_boxes = self._final_inference()
        self.fast_events, self.fast_classes, self.fast_scores, \
        self.fast_boxes = self._fast_inference()
        self.last_batch_performance = self._eval_performance()

    # =================== Parsing functions for tensorflow records
    def parser_input_function(self, example_proto):
        """Input function to decode TFrecords and matching of dboxes.
        Data augmentation and matching with default boxes is performed.

        :param example_proto: single element of the batch for read in
        :return : a dict of elements to feed the network.
        """
        frame, events, xcent, ycent, w, h, gt_labels = decode_tfrecord(
                example_proto)
        frame, events, xcent, ycent, w, h = tf.cond(
                self.is_training,
                lambda: self.augmenter.augment(frame, events, xcent,
                                               ycent, w, h),
                lambda: (frame, events, xcent, ycent, w, h))
        y_true_conf, y_true_conf_score, y_true_loc = ssd_common.match_gt_to_db(
                xcent, ycent, w, h, gt_labels, self._dboxes_lookup_dict,
                self.opt.iou_thresh,
                self._params.prior_scaling)
        gt_labels, gt_boxes = ssd_common.format_gt(xcent, ycent, w, h,
                                                   gt_labels, self.opt.gt_size)
        return {"frame":           frame,
                "events":          events,
                "y_db_conf":       y_true_conf,
                "y_db_conf_score": y_true_conf_score,
                "y_db_loc":        y_true_loc,
                "gt_labels":       gt_labels,
                "gt_boxes":        gt_boxes
                }

    # =================== Functional definition of SSD Davis network
    def _inference(self, dropout_keep_prob=0.35, weight_decay=0.0001,
                   scope='ssd_davis'):
        """
        Forward pass through the network, input through placeholder: 
        self.events_in
        expected scaled between [-1, 1]
        :param dropout_keep_prob: dropout probability
        :param weight_decay: default weight decay for regularization
        :param scope: scope_name
        :return: raw predictions for each default box: both classification & 
        localization
        """
        # End_points collect relevant activations for external use.
        end_points = {}
        with tf.variable_scope(scope, 'davis_net_core',
                               regularizer=tf.contrib.layers.l2_regularizer(
                                       weight_decay),
                               initializer=tf.contrib.layers.xavier_initializer()):
            # Prescaling
            # nope
            # Block 1.
            net = tf.layers.conv2d(self.events_in, 64, [3, 3], strides=1,
                                   use_bias=False,
                                   activation=None, padding='SAME',
                                   name="conv_1_1")
            net = tf.layers.conv2d(net, 64, [3, 3], strides=2,
                                   activation=tf.nn.elu, padding='VALID',
                                   name="conv_1_2")
            net = tf.layers.max_pooling2d(net, [2, 2], strides=1,
                                          padding='VALID', name='pool1')
            end_points['block1'] = net

            # Block 2.
            net = tf.layers.conv2d(net, 64, [3, 3], strides=1, use_bias=False,
                                   activation=None, padding='VALID',
                                   name="conv_2_1")
            net = tf.nn.lrn(net)
            net = tf.layers.conv2d(net, 64, [3, 3], strides=1,
                                   activation=tf.nn.crelu, padding='VALID',
                                   name="conv_2_2")
            net = tf.layers.max_pooling2d(net, [2, 2], strides=2,
                                          padding='VALID', name='pool2')
            end_points['block2'] = net

            # Block 3.
            net = tf.layers.dropout(net, rate=dropout_keep_prob,
                                    training=self.is_training, name="dropout_3")
            net = tf.layers.conv2d(net, 256, [3, 3], strides=1, use_bias=False,
                                   activation=None, padding='VALID',
                                   name="conv_3_1")
            net = tf.layers.conv2d(net, 128, [3, 5], strides=1,
                                   activation=tf.nn.crelu, padding='VALID',
                                   name="conv_3_2")
            net = tf.layers.max_pooling2d(net, [2, 2], strides=2,
                                          padding='VALID', name='pool3')
            end_points['block3'] = net

            # Block 4.
            # Local response normalization
            # net = tf.nn.lrn(net)
            # Padding
            net = tf.pad(net, [[0, 0], [2, 2], [1, 1], [0, 0]], mode='CONSTANT',
                         name="padding_4_0", constant_values=0)
            net = tf.layers.conv2d(net, 128, 1, strides=1, padding="VALID",
                                   name="conv_4_1")
            net = tf.layers.conv2d(net, 256, [3, 5], dilation_rate=2,
                                   activation=None, padding='VALID',
                                   name="conv_4_2")
            # ====================== Feature map
            net = tf.nn.lrn(net)
            end_points['block4'] = net
            net = tf.layers.conv2d(net, 128, 1, strides=1, padding="VALID",
                                   name="conv_5_1")
            net = tf.layers.conv2d(net, 256, [3, 3], strides=2, padding="SAME",
                                   name="conv_5_2")
            # ====================== Feature map
            net = tf.nn.lrn(net)
            end_points['block5'] = net
            net = tf.layers.conv2d(net, 64, 1, strides=1, padding="VALID",
                                   name="conv_6_1")
            net = tf.layers.conv2d(net, 128, [3, 3], strides=2, padding="SAME",
                                   name="conv_6_2")
            # ====================== Feature map
            net = tf.nn.lrn(net)
            end_points['block6'] = net
            net = tf.layers.conv2d(net, 64, 1, strides=1, padding="VALID",
                                   name="conv_7_1")
            net = tf.layers.conv2d(net, 128, [3, 3], strides=1, padding="VALID",
                                   name="conv_7_2")
            # ====================== Feature map
            net = tf.nn.lrn(net)
            end_points['block7'] = net
            net = tf.layers.conv2d(net, 32, 1, strides=1, padding="VALID",
                                   name="conv_8_1")
            net = tf.layers.conv2d(net, 64, [3, 3], strides=1, padding="VALID",
                                   name="conv_8_2")
            net = tf.nn.lrn(net)
            end_points['block8'] = net
        # Record summaries
        for key in end_points.keys():
            tf.summary.histogram(end_points[key].op.name + "/activations",
                                 end_points[key], collections=['training'])
        # Prediction and localisations layers.
        predicted_conf = []
        predicted_localiz = []
        for lay_name, layer in sorted(self._params.featmap_layers.items()):
            with tf.variable_scope(lay_name + '_featuremap'):
                p_conf, p_loc = self._ssd_featuremap_hook(
                        end_points[lay_name],
                        layer,
                        self._params.num_classes)
            predicted_conf.append(p_conf)
            predicted_localiz.append(p_loc)
        predicted_conf = tf.concat(predicted_conf, axis=1)
        predicted_localiz = tf.concat(predicted_localiz, axis=1)
        return predicted_conf, predicted_localiz, end_points

    @staticmethod
    def _ssd_featuremap_hook(feature_map, fm_param, n_classes):
        """Performs dboxes classification and localization for a feature map.

        :param feature_map:
        :type feature_map: test
        :param fm_param:
        :param n_classes:
        :return: predicted confidence, and predicted localization for the 
        feature map.

        """
        pred_conf = tf.layers.conv2d(feature_map, (
                len(fm_param['aspect_ratio']) * n_classes), [3, 3],
                                     padding='SAME',
                                     use_bias=False,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                             0.0005),
                                     name='conv_conf')
        tf.summary.histogram(pred_conf.op.name + '/activations', pred_conf,
                             collections=['training'])
        pred_conf = tf.contrib.layers.flatten(pred_conf)
        pred_loc = tf.layers.conv2d(feature_map,
                                    (len(fm_param['aspect_ratio']) * 4), [3, 3],
                                    padding='SAME',
                                    use_bias=False,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                            0.00005),
                                    name='conv_loc')
        tf.summary.histogram(pred_loc.op.name + '/activations', pred_loc,
                             collections=['training'])
        pred_loc = tf.contrib.layers.flatten(pred_loc)
        return pred_conf, pred_loc

    # =================== Functional definition of SSD Davis loss function
    def _build_losses(self, ):
        """
        Cost function factory: wires all the ops for the loss computation.
        :return loss: tensorflow op to compute the loss (no optimization step 
        is performed)
        """
        # Confidence loss
        with tf.name_scope("conf") as scope:
            logits = tf.reshape(self.y_pred_conf, [-1, self._dboxes_tot_n,
                                                   self._params.num_classes])
            # ==> Positives mining
            pmask = tf.where(tf.greater(self.y_db_conf, 0),
                             x=tf.ones_like(self.y_db_conf),
                             y=tf.zeros_like(self.y_db_conf))
            fpmask = tf.cast(pmask, dtype=tf.float32)
            conf_loss_all = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.y_db_conf, logits=logits)
            conf_loss_p = tf.losses.compute_weighted_loss(conf_loss_all, fpmask)
            # ==> Negatives mining
            # compute number of neg (n_neg will be the same for each example 
            # of the batch)
            n_pos = tf.reduce_sum(pmask, axis=1)
            n_neg = tf.minimum(self.opt.neg_pos_ratio * n_pos,
                               tf.shape(self.y_db_conf)[1] - n_pos)
            n_neg = tf.maximum(1, 1 + tf.cast(tf.reduce_mean(n_neg),
                                              dtype=tf.int32))
            # probability predicted for first class
            logits_softmax = tf.nn.softmax(logits, dim=-1)
            prob_background = tf.slice(logits_softmax, begin=[0, 0, 0],
                                       size=[-1, -1, 1])
            prob_background = tf.squeeze(prob_background, axis=2)
            # take as negative examples those with lowest confidence of class
            #  0 and small overlapping with gt
            neg_cond = tf.logical_and(tf.greater(self.opt.iou_neg_mining_thresh,
                                                 self.y_db_conf_score),
                                      tf.logical_not(
                                              tf.cast(pmask, dtype=tf.bool)))
            nmask_scores = tf.where(neg_cond, prob_background,
                                    tf.ones_like(prob_background))
            top_k, _ = tf.nn.top_k(tf.negative(nmask_scores), k=n_neg,
                                   sorted=True)
            neg_thresh = tf.reduce_max(tf.negative(top_k), axis=1,
                                       keep_dims=True)
            nmask_1 = tf.logical_and(
                    tf.logical_not(tf.cast(pmask, dtype=tf.bool)),
                    nmask_scores < neg_thresh)
            fnmask = tf.cast(nmask_1, dtype=tf.float32)
            conf_loss_n = tf.losses.compute_weighted_loss(conf_loss_all, fnmask)
            # sum confidence losses for positive and negative examples
            conf_loss = tf.reduce_sum(conf_loss_p + conf_loss_n)
            tf.summary.scalar(scope, conf_loss,
                              collections=['training', 'validation'])

        # Localization loss (smooth L1 loss)
        with tf.name_scope("local") as scope:
            diff = tf.subtract(self.y_db_loc, self.y_pred_loc)
            loc_loss_l2 = tf.multiply(0.5, tf.pow(diff, 2.0))
            loc_loss_l1 = tf.subtract(tf.abs(diff), 0.5)
            smooth_l1_condition = tf.less(tf.abs(diff), 1.0)
            loc_loss = tf.where(smooth_l1_condition, x=loc_loss_l2,
                                y=loc_loss_l1)
            # loc_loss_mask is analogous to conf_loss_mask (only positives 
            # examples!)
            loc_loss_mask = fpmask
            # [0, 1, 1] -> [[[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]], ...]
            loc_loss_mask = tf.stack([loc_loss_mask] * 4, axis=2)
            # removing the inner-most dimension of above
            loc_loss_mask = tf.reshape(loc_loss_mask, [-1, self._predloc_tot_n])
            loc_loss = tf.losses.compute_weighted_loss(loc_loss, loc_loss_mask)
            loc_loss = tf.reduce_sum(loc_loss)
            loc_loss = tf.multiply(self.opt.loc_loss_weight, loc_loss)
            tf.summary.scalar(scope, loc_loss,
                              collections=['training', 'validation'])

        # Final loss: class confidence + localization regression
        with tf.name_scope("loss") as scope:
            loss = tf.add_n([conf_loss, loc_loss, tf.reduce_sum(
                    tf.losses.get_regularization_losses())])
            tf.summary.scalar(scope, loss,
                              collections=['training', 'validation'])

        # Record summaries of magnitude of gradients activation (DOESN'T 
        # apply optimization step though)
        with tf.name_scope("gradients"):
            for key in self.end_points.keys():
                grad_wrt_activations, = tf.gradients(loss, self.end_points[key])
                tf.summary.histogram("ActivGrad_" + key, grad_wrt_activations,
                                     collections=['training'])

        return loss

    # =================== Functional definition of SSD Davis optimization 
    # process
    def _build_optimizer(self, ):
        """	Set up the optimization routine

        @return: optimization step handle
        """
        with tf.name_scope("train"):
            learn_rate = tf.train.exponential_decay(
                    self.opt.learning_rate, decay_rate=0.95,
                    decay_steps=75000, global_step=self.global_step)
            # Binding loss to optimizer
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                if self.opt.optimizer_name == "Adam":
                    return tf.train.AdamOptimizer(
                            learning_rate=learn_rate,
                            epsilon=self.opt.minimizer_eps).minimize(self.loss,
                                                                     global_step=self.global_step)
                elif self.opt.optimizer_name == "RMSProp":
                    return tf.train.RMSPropOptimizer(
                            learning_rate=learn_rate,
                            epsilon=self.opt.minimizer_eps,
                            decay=0.9,
                            momentum=0.1).minimize(
                            self.loss, global_step=self.global_step)
                else:
                    raise ValueError("The selected optimizer is unknown!")

    # =================== Utility building functions
    def _build_dboxes_table(self, ):
        """ Build a look up table of all the default boxes for fast encoding 
        and decoding.
        It returns two different format of the same thing.

       :returns :
            dboxes_dict['layer'] = [(x_center, y_center, w, h), (x_center, 
            y_center, w, h),...]
            dboxes_tensor = [dboxes_n, 4] with format (x_center, y_center, w, h)
        """
        assert self._params.input_format == 'HWC'
        dboxes_dict = {}
        n_fm_layers = len(self._params.featmap_layers.keys())
        scale_span = self._params.featmap_scales_range[1] - \
                     self._params.featmap_scales_range[0]
        n_layer = 0
        for lay_name, layer in sorted(self._params.featmap_layers.items()):
            centers = [
                [(x + 0.5) / layer['size'][0], (y + 0.5) / layer['size'][1]]
                for x in range(layer['size'][0]) for y in
                range(layer['size'][1])]
            dboxes_dict.update({lay_name: []})
            for cent in centers:
                for a_idx, a in enumerate(layer['aspect_ratio']):
                    scale = self._params.featmap_scales_range[0] + \
                            scale_span / (n_fm_layers - 1) * n_layer
                    # special scale for second aspect_ratio = 1
                    if a == 1 and a_idx == 1:
                        scale = np.sqrt(scale *
                                        (self._params.featmap_scales_range[0] +
                                         scale_span / (n_fm_layers - 1) * (
                                                 n_layer + 1)))

                    # double step to enforce the dboxes to be on the image frame
                    w = scale * np.sqrt(a) / self._params.aspectratio_bias
                    h = scale / np.sqrt(a)
                    xymin_xymax = [
                        max(0, cent[0] - w / 2.),
                        max(0, cent[1] - h / 2.),
                        min(1, cent[0] + w / 2.),
                        min(1, cent[1] + h / 2.)]
                    dboxes_dict[lay_name].append(tf.constant([
                        xymin_xymax[0] + (
                                (xymin_xymax[2] - xymin_xymax[0]) / 2.),
                        xymin_xymax[1] + (
                                (xymin_xymax[3] - xymin_xymax[1]) / 2.),
                        xymin_xymax[2] - xymin_xymax[0],
                        xymin_xymax[3] - xymin_xymax[1]
                    ], dtype=tf.float32))
            n_layer += 1
        # Also format as a unique tensor
        dboxes_loc_aslist = []
        for lay_name, dboxes in sorted(dboxes_dict.items()):
            dboxes_loc_aslist += dboxes
        dboxes_tensor = tf.stack(dboxes_loc_aslist, axis=0)
        return dboxes_dict, dboxes_tensor

    # ====== Real inference for testing, input: events -> output: image with 
    # boxes
    def _decode_predictions(self, ):
        """ Takes the raw output of the network and formats it to absolute 
        coordinates

        :return classes: [batch; dboxes_n]
        :rtype classes: tensor
        :return scores: [batch; dboxes_n]
        :rtype scores: tensor
        :return boxes: [batch; dboxes_n; 4](box format [ymin, xmin, ymax, xmax])
        :rtype boxes: tensor
        """
        # todo check return doc for multiple items
        with tf.name_scope('decode_predictions'):
            logits = tf.reshape(self.y_pred_conf, [-1, self._dboxes_tot_n,
                                                   self._params.num_classes])
            probs_all = tf.nn.softmax(logits, dim=-1)
            probs_classes = tf.slice(probs_all, begin=[0, 0, 1], size=[-1, -1,
                                                                       self._params.num_classes -
                                                                       1])
            classes = tf.argmax(probs_classes, axis=2) + 1
            scores = tf.reduce_max(probs_classes, axis=2)
            loc_all = tf.reshape(self.y_pred_loc,
                                 shape=[-1, self._dboxes_tot_n, 4])
            xreal, yreal, wreal, hreal = ssd_common.decode_db(loc_all,
                                                              self._dboxes_lookup_tensor,
                                                              self._params
                                                              .prior_scaling)
            xmin, ymin, xmax, ymax = ssd_common.bboxes_center_format_to_minmax(
                    xreal, yreal, wreal, hreal)
            boxes = tf.stack([ymin, xmin, ymax, xmax], axis=2)
        return classes, scores, boxes

    def _final_inference(self, ):
        """ Runs nms across batches

        :returns : events and final output of classes, scores, boxes
        """
        with tf.name_scope('final_inference'):
            def single_batch_nms(classes_in, scores_in, boxes_in):
                _, indices_to_keep = tf.nn.top_k(scores_in, k=self.opt.pd_size)
                # extract corresponding values and boxes
                classes_in = tf.gather(classes_in, indices_to_keep)
                scores_in = tf.gather(scores_in, indices_to_keep)
                boxes_in = tf.gather(boxes_in, indices_to_keep)
                indexes_to_display = tf.image.non_max_suppression(
                        boxes=boxes_in, scores=scores_in,
                        max_output_size=self.opt.nms_maxN,
                        iou_threshold=self.opt.nms_iou_thresh)
                zero_size = self.opt.gt_size - tf.shape(indexes_to_display)[0]
                classes_in = tf.gather(classes_in, indexes_to_display)
                classes_in = tf.concat(
                        [classes_in, tf.zeros([zero_size], dtype=tf.int64)],
                        axis=0)
                scores_in = tf.gather(scores_in, indexes_to_display)
                scores_in = tf.concat(
                        [scores_in, tf.zeros([zero_size], dtype=tf.float32)],
                        axis=0)
                boxes_in = tf.gather(boxes_in, indexes_to_display)
                boxes_in = tf.concat(
                        [boxes_in, tf.zeros(tf.stack([zero_size, 4], axis=0),
                                            dtype=tf.float32)], axis=0)
                return classes_in, scores_in, boxes_in

            # map fun across batches
            res_cla, res_sco, res_box = tf.map_fn(
                    lambda x: single_batch_nms(x[0], x[1], x[2]),
                    (self.classes_pred, self.scores_pred, self.boxes_pred),
                    back_prop=False, parallel_iterations=10, swap_memory=True)
            # format the image for display
            events = tf.cast((self.events_in * 255 / 2) + (255 / 2),
                             dtype=tf.uint8)
            events = tf.concat([events] * 3, axis=3)
        return events, res_cla, res_sco, res_box

    def _fast_inference(self, ):
        """ Runs nms for a batch of dimension 1

        :returns: events and final output of classes, scores, boxes
        """
        with tf.name_scope('fast_inference'):
            # todo we can still make it faster using 'ad hoc' decode predictions
            # squeeze out single batch dimension
            classes = tf.squeeze(self.classes_pred, axis=0)
            scores = tf.squeeze(self.scores_pred, axis=0)
            boxes = tf.squeeze(self.boxes_pred, axis=0)
            events = tf.squeeze(self.events_in, axis=0)
            # confidence cutoff
            indexes_conf_cutoff = tf.where(
                    tf.greater_equal(scores, self.conf_thresh_cutoff))
            classes = tf.gather_nd(classes, indexes_conf_cutoff)
            scores = tf.gather_nd(scores, indexes_conf_cutoff)
            boxes = tf.gather_nd(boxes, indexes_conf_cutoff)
            # nms
            indexes_nms = tf.image.non_max_suppression(
                    boxes=boxes, scores=scores,
                    max_output_size=self.opt.nms_maxN,
                    iou_threshold=self.opt.nms_iou_thresh)
            classes = tf.gather(classes, indexes_nms)
            scores = tf.gather(scores, indexes_nms)
            boxes = tf.gather(boxes, indexes_nms)
            # format the image for display
            events = tf.cast((events + 1) * 255 / 2, dtype=tf.uint8)
            events = tf.concat([events] * 3, axis=2)
        return events, classes, scores, boxes

    def _eval_performance(self, ):
        # todo implement mAP use my metrics!!
        n_gt, tp, fp = metrics_tf.gather_stats_on_batch(
                self.final_classes, self.final_scores, self.final_boxes,
                self.gt_labels, self.gt_boxes, self._params.num_classes,
                self.conf_thresh_cutoff)
        acc, rec = metrics_tf.compute_acc_rec(n_gt, tp, fp)
        acc_mean = sum(acc.values()) / len(acc.keys())
        rec_mean = sum(rec.values()) / len(rec.keys())
        return {'n_gt':     n_gt,
                'tp':       tp,
                'fp':       fp,
                'acc':      acc,
                'acc_mean': acc_mean,
                'rec':      rec,
                'rec_mean': rec_mean
                }

    # Some getters and setters
    @property
    def dboxes_tot_n(self):
        return self._dboxes_tot_n

    @property
    def dboxes_lookup_tensor(self):
        return self._dboxes_lookup_tensor

    @property
    def parameters(self):
        return self._params


if __name__ == "__main__":
    # Don't run this file as main unless for debugging !
    print("Just for testing, this file should not be run as main function !")
    ssd_net = SSDNet()
    print("End of testing")
