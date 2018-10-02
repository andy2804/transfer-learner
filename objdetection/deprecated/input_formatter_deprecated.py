import numpy as np


class EventsTransform:
    def __init__(self, time_span=0.05,
                 height=180, width=240):
        self._time_span = time_span
        self._height = height
        self._width = width

    @property
    def time_span(self):
        return self._time_span

    @time_span.setter
    def time_span(self, value):
        if value < 0:
            raise ValueError(
                    "The time span to accumulate events cannot be negative!")
        self._time_span = value

    @staticmethod
    def crop_events_of_interest(events_d, frame_ts, previous_ts):
        new_events = {}
        # Crop events of interest
        idx_pos = np.logical_and(np.greater(frame_ts, events_d["pos"][:, 0]),
                                 np.greater_equal(events_d["pos"][:, 0],
                                                  previous_ts))
        idx_neg = np.logical_and(np.greater(frame_ts, events_d["neg"][:, 0]),
                                 np.greater_equal(events_d["neg"][:, 0],
                                                  previous_ts))
        new_events["pos"] = events_d["pos"][idx_pos]
        new_events["neg"] = events_d["neg"][idx_neg]
        return new_events, idx_pos, idx_neg


class SAE(EventsTransform):
    def __init__(self, mode, time_span=0.05):
        super().__init__(time_span)
        if 'exp' in mode:
            self.weight_fun = lambda x: np.exp(np.multiply(x, 3))
        elif 'gaus' in mode:
            self.weight_fun = lambda x: np.exp(
                    - np.power(np.multiply(x, 2.5), 2))
        elif 'linear' in mode:
            self.weight_fun = lambda x: 1 - x
        else:
            raise ValueError("Unknown formatting method!")

        if '01' in mode:
            self.crop_and_format_events = self.zeroone
        elif '_11' in mode:
            self.crop_and_format_events = self.minusoneone
        else:
            raise ValueError("Unknown formatting method!")

    def zeroone(self, events_d, frame_ts, previous_ts):
        new_events, _, _ = self.crop_events_of_interest(events_d, frame_ts,
                                                        previous_ts)
        delta_ts = frame_ts - previous_ts
        # Build SAE frame
        ndarray = np.zeros((self._height, self._width, 2), dtype=np.float32)
        for key in new_events.keys():
            for event in new_events[key]:
                ts, x, y, p = event
                x, y, p = int(x), int(y), int(p)
                # Values scaled  between 0 and -1 with a weighting function
                value = self.weight_fun(np.divide(ts - frame_ts, delta_ts))
                ndarray[y, x, p] = value
        ndarray = np.amax(ndarray, axis=-1)
        return ndarray.astype(np.float)

    def minusoneone(self, events_d, frame_ts, previous_ts):
        new_events, _, _ = self.crop_events_of_interest(events_d, frame_ts,
                                                        previous_ts)
        delta_ts = frame_ts - previous_ts
        # Build SAE frame
        ndarray = np.zeros((self._height, self._width, 2), dtype=np.float32)
        for key in new_events.keys():
            for event in new_events[key]:
                ts, x, y, p = event
                x, y, p = int(x), int(y), int(p)
                # Values scaled between 0 and -1 with a weighting function
                value = self.weight_fun(np.divide(ts - frame_ts, delta_ts))
                ndarray[y, x, p] = value
        neg_events = ndarray[:, :, 0]
        pos_events = ndarray[:, :, 1]
        cond = np.where(neg_events > pos_events)
        events_imag = pos_events
        events_imag[cond] = - neg_events[cond]
        return events_imag.astype(np.float)


class THREED(EventsTransform):
    def __init__(self, time_span=0.05):
        super().__init__(time_span)

    # todo input for network(this is only for plotting)
    def crop_and_format_events(self, events_d, frame_ts, previous_ts,
                               time_bins):
        _, idx_pos, idx_neg = self.crop_events_of_interest(events_d, frame_ts,
                                                           previous_ts)
        # Build 3D array
        channels = 2
        time_bins = np.linspace(previous_ts, frame_ts, num=time_bins,
                                endpoint=True)
        time_quant_pos = np.subtract(
                np.digitize(events_d["pos"][idx_pos, 0], time_bins,
                            right=False),
                1).astype(int)
        time_quant_neg = np.subtract(
                np.digitize(events_d["neg"][idx_neg, 0], time_bins,
                            right=False),
                1).astype(int)
        data = np.zeros((time_bins.size, 180, 240, channels),
                        dtype=np.int8)  # todo generalize size

        data[time_quant_pos, (events_d["pos"][idx_pos, 2]).astype(int),
             (events_d["pos"][idx_pos, 1]).astype(int), 0] = 1
        data[time_quant_neg, (events_d["neg"][idx_neg, 2]).astype(int),
             (events_d["neg"][idx_neg, 1]).astype(int), channels - 1] = 1
        return data

    def crop_and_format_frames(self, frames_d, frame_ts, previous_ts,
                               time_bins):
        frames_list = [[f[0], f[1]] for f in frames_d if
                       frame_ts > f[1] > previous_ts]
        time_bins = np.linspace(previous_ts, frame_ts, num=time_bins,
                                endpoint=True)
        for frame in frames_list:
            frame[1] = np.subtract(
                    np.digitize(frame[1], time_bins, right=False), 1)
        return frames_list


class SUM(EventsTransform):
    def __init__(self, mode, time_span=0.05):
        super().__init__(time_span)
        if 'plain_sum' in mode:
            self.crop_and_format_events = self.plainsum
        elif 'saturated_sum' in mode:
            self.crop_and_format_events = self.satsum
        else:
            raise ValueError("Unknown formatting method!")

    def plainsum(self, events_d, frame_ts, previous_ts):
        # todo input for 2d net, documentation
        new_events, _, _ = self.crop_events_of_interest(events_d, frame_ts,
                                                        previous_ts)
        ndarray = np.zeros((180, 240, 2), dtype=np.float32)
        for key in new_events.keys():
            for event in new_events[key]:
                ts, x, y, p = event
                x, y, p = int(x), int(y), int(p)
                ndarray[y, x, p] += 1
        # rescaling between -1 and 1
        ndarray_pos = np.divide(ndarray[:, :, 1], np.max(ndarray[:, :, 1]))
        ndarray_neg = np.divide(ndarray[:, :, 0], np.max(ndarray[:, :, 0]))
        return ndarray_pos - ndarray_neg

    def satsum(self, events_d, frame_ts, previous_ts):
        # todo input for 2d net, documentation
        new_events, _, _ = self.crop_events_of_interest(events_d, frame_ts,
                                                        previous_ts)
        ndarray = np.zeros((180, 240, 2), dtype=np.float32)
        for key in new_events.keys():
            for event in new_events[key]:
                ts, x, y, p = event
                x, y, p = int(x), int(y), int(p)
                ndarray[y, x, p] = 1
        # values betwen -1 and 1
        ndarray = ndarray[:, :, 1] - ndarray[:, :, 0]
        return ndarray
