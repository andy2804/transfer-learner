"""
author: az
"""
from functools import wraps

import numpy as np
import tensorflow as tf

from objdetection.rgb2events.nets import ssd_common
from objdetection.rgb2events.nets import ObjectDetector
from objdetection.meta.datasets import DataAugSae
from objdetection.meta.datasets.encoder_tfrecord_rosbag import EncoderTFrecRosbag
from objdetection.meta.utils_events import Sae
from objdetection.meta.utils_generic import magic_constants


# ====== Decorator for tf namescope
def tf_scope(scope):
    def tf_scope_decorator(func):
        @wraps(func)
        def func_wrapper(*args, **kargs):
            with tf.name_scope(scope):
                return func(*args, **kargs)

        return func_wrapper

    return tf_scope_decorator


# ======


class SSDnet(ObjectDetector, EncoderTFrecRosbag, Sae, DataAugSae):

    def __init__(self, objdet_par, net_par, learn_par, events_transform_par):
        ObjectDetector.__init__(self, objdet_par)
        EncoderTFrecRosbag.__init__(self, )
        Sae.__init__(self, events_transform_par.weight_fn, events_transform_par.time_window)
        DataAugSae.__init__(self, )

        self._net_par, self._learn_par = None, None
        self.net_par, self.learn_par = net_par, learn_par
        self._dboxes_lookup_dict, self._dboxes_lookup_tensor = self._build_dboxes_lookup()
        # options
        self._fm_cells_tot_n = int(sum([v["size"][0] * v["size"][1]
                                        for v in self.net_par.featmap_layers.values()]))
        self._dboxes_tot_n = int(sum([v["size"][0] * v["size"][1] * len(v["aspect_ratio"])
                                      for v in self.net_par.featmap_layers.values()]))
        self._predconf_tot_n = self._dboxes_tot_n * self.objdet_par.num_classes
        self._predloc_tot_n = self._dboxes_tot_n * 4

        # fixme all the following into a function
        # ============= Placeholders for the model inputs
        # self.global_step = tf.Variable(0, trainable=False)
        self.events_in = tf.placeholder(
                tf.float32, shape=self.objdet_par.input_shape, name='events_in')
        # self.y_db_label = tf.placeholder(tf.int32, [None, self._dboxes_tot_n],
        #                                  name='y_db_conf')
        # self.y_db_iou_score = tf.placeholder(tf.float32, [None, self._dboxes_tot_n],
        #                                      name='y_db_conf')
        # self.y_db_loc = tf.placeholder(tf.float32, [None, self._predloc_tot_n],
        #                                name='y_db_loc')  # localization
        self.y_db_label, self.y_db_iou_sc, self.y_db_loc = None, None, None
        # Global step
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        # Init output handels: Real initialization happens from get next batch
        self.y_pred_label, self.y_pred_loc, self.end_points = None, None, None
        self.loss = None
        self.train_opt = None
        self.pred_label, self.pred_label_scores, self.pred_boxes = None, None, None
        # Placeholder for switching easly between training and validation
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        # Placeholders for performance metrics?!
        self.gt_boxes = tf.placeholder(tf.float32, [None, self.net_par.gt_size, 4])
        self.gt_labels = tf.placeholder(tf.int64, [None, self.net_par.gt_size])

    def parse_input(self, input_proto):
        """Input function to decode TFrecords and matching of dboxes.
        Data augmentation and matching with default boxes is performed.

        :param input_proto: single element of the batch for read in
        :returns : a dict of elements to feed the network.
        """
        input_proto_dict = self.decode(input_proto)
        input_proto_dict['events'] = self.tf_transform_events(
                input_proto_dict['events'], to_ts=input_proto_dict['events'][-1, 0])
        """
        input_proto_dict = tf.cond(
                self.is_training,
                lambda: self._data_augmenter.augment(input_proto_dict),
                lambda: input_proto_dict)
        """
        y_db_label, y_db_iou_score, y_db_loc = \
            ssd_common.match_gt_to_db(
                    input_proto_dict["xcent"],
                    input_proto_dict["ycent"],
                    input_proto_dict["w"],
                    input_proto_dict["h"],
                    input_proto_dict["gt_labels"],
                    self._dboxes_lookup_dict,
                    self.net_par.iou_thresh,
                    self.net_par.prior_scaling)

        gt_labels, gt_boxes = \
            ssd_common.format_gt(input_proto_dict["xcent"],
                                 input_proto_dict["ycent"],
                                 input_proto_dict["w"],
                                 input_proto_dict["h"],
                                 input_proto_dict["gt_labels"],
                                 self.net_par.gt_size)

        return {"frame":          input_proto_dict["frame"],
                "events":         input_proto_dict["events"],
                "y_db_label":     y_db_label,
                "y_db_iou_score": y_db_iou_score,
                "y_db_loc":       y_db_loc,
                "gt_labels":      gt_labels,
                "gt_boxes":       gt_boxes
                }

    def forwardpass(self, dropout_keep_prob=0.35, weight_decay=0.0001, scope='ssd_davis'):
        """
        Forward pass through the network, input through placeholder:
        self.events_in
        expected scaled between [-1, 1]
        :param dropout_keep_prob: dropout probability
        :param weight_decay: default weight decay for regularization
        :param scope: scope_name
        :return: raw predictions for each default box: [y_pred_label, y_pred_loc, end_points]
        """
        # End_points collect relevant activations for external use.
        end_points = {}
        with tf.variable_scope(
                scope, 'davis_net_core',
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                initializer=tf.contrib.layers.xavier_initializer()):
            # Prescaling
            # nope
            # Block 1.
            net = tf.layers.conv2d(self.events_in, 64, [3, 3], strides=1,
                                   use_bias=False, activation=None, padding='SAME',
                                   name="conv_1_1")
            net = tf.layers.conv2d(net, 64, [3, 3], strides=2,
                                   activation=tf.nn.elu, padding='VALID',
                                   name="conv_1_2")
            net = tf.layers.max_pooling2d(net, [2, 2], strides=1,
                                          padding='VALID', name='pool1')
            end_points['block1'] = net

            # Block 2.
            net = tf.layers.conv2d(net, 64, [3, 3], strides=1, use_bias=False,
                                   activation=None, padding='VALID', name="conv_2_1")
            net = tf.nn.lrn(net)
            net = tf.layers.conv2d(net, 64, [3, 3], strides=1,
                                   activation=tf.nn.crelu, padding='VALID', name="conv_2_2")
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
                                   activation=tf.nn.crelu, padding='VALID', name="conv_3_2")
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
        y_pred_label = []
        y_pred_loc = []
        for lay_name, layer in sorted(self.net_par.featmap_layers.items()):
            with tf.variable_scope(lay_name + '_featuremap'):
                p_conf, p_loc = self._ssd_featuremap_hook(
                        end_points[lay_name],
                        layer,
                        self.objdet_par.num_classes)
            y_pred_label.append(p_conf)
            y_pred_loc.append(p_loc)
        y_pred_label = tf.concat(y_pred_label, axis=1)
        y_pred_loc = tf.concat(y_pred_loc, axis=1)
        return y_pred_label, y_pred_loc, end_points

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
        pred_conf = tf.layers.conv2d(
                feature_map,
                (len(fm_param['aspect_ratio']) * n_classes),
                [3, 3],
                padding='SAME',
                use_bias=False,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0005),
                name='conv_conf')
        tf.summary.histogram(pred_conf.op.name + '/activations', pred_conf,
                             collections=['training'])
        pred_conf = tf.contrib.layers.flatten(pred_conf)
        pred_loc = tf.layers.conv2d(
                feature_map,
                (len(fm_param['aspect_ratio']) * 4), [3, 3],
                padding='SAME',
                use_bias=False,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.00005),
                name='conv_loc')
        tf.summary.histogram(pred_loc.op.name + '/activations', pred_loc,
                             collections=['training'])
        pred_loc = tf.contrib.layers.flatten(pred_loc)
        return pred_conf, pred_loc

    @tf_scope('losses')
    def tower_losses(self):
        """
        Cost function factory: wires all the ops for the loss
        computation.
        :return loss: tensorflow op to compute the loss (no
        optimization step
        is performed)
        """
        # Confidence loss
        with tf.name_scope("conf") as scope:
            logits = tf.reshape(
                    self.y_pred_label, [-1, self._dboxes_tot_n, self.objdet_par.num_classes])
            # ==> Positives mining
            pmask_bool = tf.where(tf.greater(self.y_db_label, 0),
                                  x=tf.ones_like(self.y_db_label),
                                  y=tf.zeros_like(self.y_db_label))
            pmask_float = tf.cast(pmask_bool, dtype=tf.float32)
            conf_loss_all = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.y_db_label, logits=logits)
            conf_loss_p = tf.losses.compute_weighted_loss(
                    conf_loss_all, pmask_float)
            # ==> Negatives mining
            # compute number of neg (n_neg will be the same for each example of the batch)
            n_pos = tf.reduce_sum(pmask_bool, axis=1)
            n_neg = tf.minimum(self.net_par.neg_pos_ratio * n_pos,
                               tf.shape(self.y_db_label, out_type=tf.int64)[1] - n_pos)
            n_neg = tf.maximum(1, 1 + tf.cast(tf.reduce_mean(n_neg),
                                              dtype=tf.int32))
            # probability predicted for zero class (background)
            logits_softmax = tf.nn.softmax(logits, axis=-1)
            prob_background = tf.slice(logits_softmax, begin=[0, 0, 0],
                                       size=[-1, -1, 1])
            prob_background = tf.squeeze(prob_background, axis=2)
            # take as negative examples those with lowest confidence of class
            #  0 and small overlapping with gt
            neg_cond = tf.logical_and(
                    tf.greater(self.net_par.iou_neg_mining_thresh,
                               self.y_db_iou_score),  # fixme double-check
                    tf.logical_not(tf.cast(pmask_bool, dtype=tf.bool)))
            nmask_scores = tf.where(neg_cond, prob_background,
                                    tf.ones_like(prob_background))
            top_k, _ = tf.nn.top_k(tf.negative(nmask_scores), k=n_neg, sorted=True)
            neg_thresh = tf.reduce_max(tf.negative(top_k), axis=1, keepdims=True)
            nmask_1 = tf.logical_and(
                    tf.logical_not(tf.cast(pmask_bool, dtype=tf.bool)),
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
            loc_loss = tf.where(smooth_l1_condition, x=loc_loss_l2, y=loc_loss_l1)
            # loc_loss_mask is analogous to conf_loss_mask (only positives
            # examples!)
            loc_loss_mask = pmask_float
            # [0, 1, 1] -> [[[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]], ...]
            loc_loss_mask = tf.stack([loc_loss_mask] * 4, axis=2)
            # removing the inner-most dimension of above
            loc_loss_mask = tf.reshape(loc_loss_mask, [-1, self._predloc_tot_n])
            loc_loss = tf.losses.compute_weighted_loss(loc_loss, loc_loss_mask)
            loc_loss = tf.reduce_sum(loc_loss)
            loc_loss = tf.multiply(self.learn_par.loc_loss_weight, loc_loss)
            tf.summary.scalar(scope, loc_loss, collections=['training', 'validation'])

        # Final loss: class confidence + localization regression
        with tf.name_scope("total") as scope:
            loss = tf.add_n([conf_loss, loc_loss, tf.reduce_sum(
                    tf.losses.get_regularization_losses())])
            tf.summary.scalar(scope, loss, collections=['training', 'validation'])

        # Record summaries of magnitude of gradients activation (DOESN'T
        # apply optimization step though)
        with tf.name_scope("gradients"):
            for key in self.end_points.keys():
                grad_wrt_activations, = tf.gradients(loss, self.end_points[key])
                tf.summary.histogram("ActivGrad_" + key, grad_wrt_activations,
                                     collections=['training'])
        tf.losses.add_loss(loss)  # fixme
        return loss

    @tf_scope('optimizer')
    def optimizer(self):
        # fixme eliminate hard coded variables
        learn_rate = tf.train.exponential_decay(
                self.learn_par.learning_rate, decay_rate=0.95,
                decay_steps=75000, global_step=tf.train.get_global_step())
        # Binding loss to optimizer
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            if self.learn_par.optimizer_name == "Adam":
                return tf.train.AdamOptimizer(
                        learning_rate=learn_rate,
                        epsilon=self.learn_par.minimizer_eps
                ).minimize(self.loss)
            elif self.learn_par.optimizer_name == "RMSProp":
                return tf.train.RMSPropOptimizer(
                        learning_rate=learn_rate,
                        epsilon=self.learn_par.minimizer_eps,
                        decay=0.9,  # fixme move constant
                        momentum=0.1)  # fixme move constant
            else:
                raise ValueError("The selected optimizer is unknown!")

    """
    @tf_scope('nms')
    def nms(self):
        if self.objdet_par.real_time:
            return self._fast_inference()
        else:
            return self._final_inference()
    """

    def _build_dboxes_lookup(self):
        """ Build a look up table of all the default boxes for fast encoding
        and decoding.
        It returns two different format of the same thing.
        :returns :  dboxes_dict['layer'] = [(x_center, y_center, w, h),
                    (x_center, y_center, w, h),...]
                    dboxes_tensor = [dboxes_n, 4] with format (x_center,
                    y_center, w, h)
        """
        assert self.objdet_par.input_format == 'HWC'
        dboxes_dict = {}
        n_fm_layers = len(self.net_par.featmap_layers.keys())
        scale_span = self.net_par.featmap_scales_range[1] - \
                     self.net_par.featmap_scales_range[0]
        layer_count = 0
        for lay_name, layer in sorted(self.net_par.featmap_layers.items()):
            # centers of dboxes for one layer[normalized coordiantes]
            centers = [
                [(x + 0.5) / layer['size'][0], (y + 0.5) / layer['size'][1]]
                for x in range(layer['size'][0]) for y in
                range(layer['size'][1])]
            dboxes_dict.update({lay_name: []})
            # foreach center add different aspect ratios and scales
            for cent in centers:
                for a_idx, a in enumerate(layer['aspect_ratio']):
                    scale = self.net_par.featmap_scales_range[0] + \
                            scale_span / (n_fm_layers - 1) * layer_count
                    # second scale for aspect ration = 1
                    if a == 1 and a_idx == 1:
                        scale = np.sqrt(
                                scale * (self.net_par.featmap_scales_range[0] +
                                         scale_span / (n_fm_layers - 1) * (
                                                 layer_count + 1)))

                    # double transformation to force the
                    # dboxes to be on the image frame
                    w = scale * np.sqrt(a) / self.net_par.aspectratio_bias
                    h = scale / np.sqrt(a)
                    xymin_xymax = [
                        max(0, cent[0] - w / 2.),
                        max(0, cent[1] - h / 2.),
                        min(1, cent[0] + w / 2.),
                        min(1, cent[1] + h / 2.)]
                    dboxes_dict[lay_name].append(tf.constant(
                            [xymin_xymax[0] + (
                                    (xymin_xymax[2] - xymin_xymax[0]) / 2.),
                             xymin_xymax[1] + (
                                     (xymin_xymax[3] - xymin_xymax[1]) / 2.),
                             xymin_xymax[2] - xymin_xymax[0],
                             xymin_xymax[3] - xymin_xymax[1]
                             ], dtype=tf.float32))
            layer_count += 1
        # Also format as a unique tensor
        dboxes_loc_aslist = []
        for lay_name, dboxes in sorted(dboxes_dict.items()):
            dboxes_loc_aslist += dboxes
        dboxes_tensor = tf.stack(dboxes_loc_aslist, axis=0)
        return dboxes_dict, dboxes_tensor

    @tf_scope('decode_predictions')
    def _decode_predictions(self, ):
        """ Takes the raw output of the network and formats it to absolute
        coordinates

        :return labels: [batch; dboxes_n]
        :rtype labels: tensor
        :return labels_score: [batch; dboxes_n]
        :rtype labels_score: tensor
        :return boxes: [batch; dboxes_n; 4](box format [ymin, xmin, ymax, xmax])
        :rtype boxes: tensor
        """
        # todo check return doc for multiple items
        logits = tf.reshape(self.y_pred_label,
                            [-1, self._dboxes_tot_n, self.objdet_par.num_classes])
        probs_all = tf.nn.softmax(logits, axis=-1)
        # cut out background class
        probs_classes = tf.slice(probs_all, begin=[0, 0, 1],
                                 size=[-1, -1, self.objdet_par.num_classes - 1])
        labels = tf.argmax(probs_classes, axis=2) + 1
        labels_score = tf.reduce_max(probs_classes, axis=2)
        loc_all = tf.reshape(self.y_pred_loc, shape=[-1, self._dboxes_tot_n, 4])
        xreal, yreal, wreal, hreal = ssd_common.decode_db(
                loc_all,
                self._dboxes_lookup_tensor,
                self.net_par.prior_scaling)
        xmin, ymin, xmax, ymax = ssd_common.bboxes_center_format_to_minmax(
                xreal, yreal, wreal, hreal)
        boxes = tf.stack([ymin, xmin, ymax, xmax], axis=2)
        return labels, labels_score, boxes

    @tf_scope('final_inference')
    def _final_inference(self, ):
        """ Runs nms across batches

        :returns : events and final output of classes, scores, boxes
        """

        def single_batch_nms(_labels, _scores, _boxes):
            _, indices_to_keep = tf.nn.top_k(_scores, k=self.net_par.pd_size)
            # extract corresponding values and boxes
            _labels = tf.gather(_labels, indices_to_keep)
            _scores = tf.gather(_scores, indices_to_keep)
            _boxes = tf.gather(_boxes, indices_to_keep)
            indexes_to_display = tf.image.non_max_suppression(
                    boxes=_boxes, scores=_scores,
                    max_output_size=self.objdet_par.nms_maxN,
                    iou_threshold=self.objdet_par.nms_iou_thresh)
            zero_size = self.net_par.gt_size - tf.shape(indexes_to_display)[0]
            _labels = tf.gather(_labels, indexes_to_display)
            _labels = tf.concat(
                    [_labels, tf.zeros([zero_size], dtype=tf.int64)], axis=0)
            _scores = tf.gather(_scores, indexes_to_display)
            _scores = tf.concat(
                    [_scores, tf.zeros([zero_size], dtype=tf.float32)], axis=0)
            _boxes = tf.gather(_boxes, indexes_to_display)
            _boxes = tf.concat(
                    [_boxes, tf.zeros(tf.stack([zero_size, 4], axis=0),
                                      dtype=tf.float32)], axis=0)
            return _labels, _scores, _boxes

        # map fun across batches
        res_cla, res_sco, res_box = tf.map_fn(
                lambda x: single_batch_nms(x[0], x[1], x[2]),
                (self.pred_label, self.pred_label_scores, self.pred_boxes),
                back_prop=False, parallel_iterations=10, swap_memory=True)
        # format the image for display
        events = tf.cast((self.events_in * 255 / 2) + (255 / 2), dtype=tf.uint8)
        events = tf.concat([events] * 3, axis=3)
        return events, res_cla, res_sco, res_box

    @tf_scope('fast_inference')
    def _fast_inference(self, ):
        """ Runs nms for a batch of dimension 1
        :returns: events and final output of classes, scores, boxes
        """
        with tf.name_scope('fast_inference'):
            # todo we can still make it faster using 'ad hoc' decode predictions
            # squeeze out single batch dimension
            classes = tf.squeeze(self.pred_label, axis=0)
            scores = tf.squeeze(self.pred_label_scores, axis=0)
            boxes = tf.squeeze(self.pred_boxes, axis=0)
            events = tf.squeeze(self.events_in, axis=0)
            # confidence cutoff
            indexes_conf_cutoff = tf.where(
                    tf.greater_equal(scores, self.objdet_par.retrival_conf_thresh))
            classes = tf.gather_nd(classes, indexes_conf_cutoff)
            scores = tf.gather_nd(scores, indexes_conf_cutoff)
            boxes = tf.gather_nd(boxes, indexes_conf_cutoff)
            # nms
            indexes_nms = tf.image.non_max_suppression(
                    boxes=boxes, scores=scores,
                    max_output_size=self.objdet_par.nms_maxN,
                    iou_threshold=self.objdet_par.nms_iou_thresh)
            classes = tf.gather(classes, indexes_nms)
            scores = tf.gather(scores, indexes_nms)
            boxes = tf.gather(boxes, indexes_nms)
            # format the image for display
            events = tf.cast((events + 1) * 255 / 2, dtype=tf.uint8)
            events = tf.concat([events] * 3, axis=2)
        return events, classes, scores, boxes

    """
    def clone_function(self):
        # todo everything
        end_points = self.forwardpass()
        self.tower_losses()
        return end_points
    """

    def next_batch(self, next_batch):
        self.events_in = next_batch["events"]
        self.y_db_label = next_batch["y_db_label"]
        self.y_db_iou_score = next_batch["y_db_iou_score"]
        self.y_db_loc = next_batch["y_db_loc"]
        self._build_forwardpass()
        self._build_losses()
        self._build_optimizer()
        self._build_pred()

    def _build_forwardpass(self, ):
        self.y_pred_label, self.y_pred_loc, self.end_points = self.forwardpass()

    def _build_losses(self, ):
        self.loss = self.tower_losses()

    def _build_optimizer(self, ):
        self.train_opt = self.optimizer()

    def _build_pred(self, ):
        self.pred_label, self.pred_label_scores, self.pred_boxes = self._decode_predictions()

    @property
    def learn_par(self):
        return self._learn_par

    @learn_par.setter
    def learn_par(self, params):
        if isinstance(params, magic_constants.LearnParams):
            self._learn_par = params
        else:
            raise ValueError('LearnParams value is not acceptable')

    @property
    def net_par(self):
        return self._net_par

    @net_par.setter
    def net_par(self, params):
        if isinstance(params, magic_constants.SSDnetParams):
            self._net_par = params
        else:
            raise ValueError('NetParams value is not acceptable')


if __name__ == '__main__':
    print("Testing only!!")
    args = {"objdet_par":           magic_constants.DEFAULT_OBJDET_PAR,
            "net_par":              magic_constants.DEFAULT_SSDnet_PARAMS,
            "learn_par":            magic_constants.DEFAULT_LEARN_PARAMS,
            "events_transform_par": magic_constants.DEFAULT_EVENT_TRANSFORM_PARAMS
            }
    ssdnet_test = SSDnet(**args)
    print("test ended")
