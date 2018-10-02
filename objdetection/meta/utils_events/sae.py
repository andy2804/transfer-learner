"""
author: az
"""
import numpy as np
import tensorflow as tf

from objdetection.meta.utils_events.events_transform import EventsTransform


class Sae(EventsTransform):
    """Surface of active events transform.
    Outputs a single frame with values between -1 and +1 """

    def __init__(self, weight_fn, time_window=5):
        """
        :type weight_fn: str
        :param time_window: [ms]
        """
        super().__init__(time_window)
        self._weight = weight_fn
        if 'exp' in weight_fn:
            self.tf_weight_fun = lambda x: tf.exp(-tf.multiply(x, 3))
            self.np_weight_fun = lambda x: np.exp(-np.multiply(x, 3))
        elif 'gaus' in weight_fn:
            self.tf_weight_fun = lambda x: tf.exp(-tf.pow(x * 2.5, 2))
            self.np_weight_fun = lambda x: np.exp(-np.power(x * 2.5, 2))
        elif 'linear' in weight_fn:
            self.tf_weight_fun = lambda x: 1 - x
            self.np_weight_fun = lambda x: 1 - x
        else:
            raise ValueError("Unknown formatting method!")

    def tf_transform_events(self, events, to_ts, keep_polarity=False):
        """
        :param keep_polarity:
        :param events: [[ts,x,y,p], [ts,x,y,p],...]
        :param to_ts:
        :returns: tensor of shape (1,h,w,ch) where ch is determined by the keep_polarity flag
        """
        from_ts = to_ts - tf.cast(self.time_span * 1e6, dtype=tf.float64)
        events = self.tf_crop_events_of_interest(events, from_ts, to_ts)
        delta_ts = to_ts - from_ts

        # Build SAE frame
        events_frame = tf.zeros([self._height, self._width, 2], dtype=tf.float32)
        num_events = tf.shape(events)[0]

        def m_condition(_i, _events, _events_frame):
            return tf.less(_i, num_events)

        def m_body(_i, _events, _events_frame):
            # messy because tf does not support indexed assignment
            event = _events[_i, :]
            # xyp ->yxp
            indices = tf.cast([tf.stack([event[2], event[1], event[3]])], dtype=tf.int64)
            value = tf.cast([self.tf_weight_fun(tf.divide(to_ts - event[0], delta_ts))],
                            dtype=tf.float32)
            update_tensor = tf.SparseTensor(
                    indices, value, tf.shape(_events_frame, out_type=tf.int64))
            update_tensor = tf.sparse_tensor_to_dense(update_tensor)
            _events_frame = tf.where(
                    tf.greater(update_tensor, _events_frame), update_tensor, _events_frame)
            return [_i + 1, _events, _events_frame]

        # loop over events
        i = 0
        [i, events, events_frame] = tf.while_loop(
                m_condition, m_body, [i, events, events_frame],
                parallel_iterations=10, back_prop=False)
        if not keep_polarity:
            neg_events = events_frame[:, :, 0]
            pos_events = events_frame[:, :, 1]
            cond = tf.greater(neg_events, pos_events)
            events_frame = tf.where(cond, -neg_events, pos_events)
        return tf.expand_dims(events_frame, axis=-1)

    def np_transform_events(self, events, to_ts, keep_polarity=False):
        """
        :param keep_polarity:
        :param events: [[ts,x,y,p], [ts,x,y,p],...]
        :param to_ts: [ms]
        """
        from_ts = to_ts - np.float64(self.time_span * 1e6)
        # Build SAE frame
        ndarray = np.zeros((self._height, self._width, 2), dtype=np.float32)

        new_events = self.np_crop_events_of_interest(events, from_ts, to_ts)
        delta_ts = to_ts - from_ts
        values = self.np_weight_fun(np.divide(to_ts - new_events[:, 0], delta_ts))
        for i, event_basic in enumerate(new_events[:, 1:].astype(np.int64).tolist()):
            x, y, p = event_basic
            ndarray[y, x, p] = values[i] if values[i] > ndarray[y, x, p] else ndarray[y, x, p]
        if not keep_polarity:
            neg_events = ndarray[:, :, 0]
            pos_events = ndarray[:, :, 1]
            cond = np.where(neg_events > pos_events)
            events_imag = pos_events
            events_imag[cond] = - neg_events[cond]
            events_imag = np.stack([(events_imag + 1) * 255 / 2] * 3, axis=-1).astype(np.uint8)
        else:
            events_imag = ndarray
        return events_imag

    def np_crazy_transform(self, events, to_ts):
        from_ts = to_ts - np.float64(self.time_span * 1e6)
        new_events = self.np_crop_events_of_interest(events, from_ts, to_ts)
        activity_frame = self._compute_activity_neigh(new_events)
        events_frame = self.np_transform_events(new_events, to_ts, keep_polarity=True)
        events_frame = np.concatenate(
                [events_frame[:, :, [0]], activity_frame, events_frame[:, :, [1]]], axis=-1)
        events_frame *= 255
        return events_frame.astype(np.uint8)

    def get_encoding_str(self):
        """
        :return method_str: string describing the encoding method used for the events
        """
        method_str = self._weight + "_" + "{:d}".format(self._time_span)
        return method_str

    def _compute_activity(self, new_events):
        """Activity at pixel location"""
        frame = np.zeros((self._height, self._width, 1), dtype=np.float)
        for e in new_events:
            y, x = int(e[2]), int(e[1])
            frame[y, x, 0] += 1
        # normalize
        return frame / np.max(frame)

    def _compute_activity_neigh(self, new_events):
        """Compute activity in the neighborhood of the event"""
        neighbor_offset = (-1, 0, +1)
        frame = np.zeros((self._height, self._width, 1), dtype=np.float)
        for e in new_events:
            y, x = int(e[2]), int(e[1])
            update_loc = [
                (max(min(y + y_o, self._height - 1), 0), (max(min(x + x_o, self._width - 1), 0)))
                for y_o in neighbor_offset for x_o in neighbor_offset if
                (y_o != 0 or x_o != 0)]
            for y, x in update_loc:
                frame[y, x, 0] += 1
        # normalize
        return frame / np.max(frame)
