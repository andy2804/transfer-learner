from objdetection.meta.utils_events.events_transform import EventsTransform


class HistogramOfActivity(EventsTransform):
    def __init__(self, mode, time_window=0.05):
        super().__init__(time_window)
        if 'plain_sum' in mode:
            self.crop_and_format_events = self.plainsum
        elif 'saturated_sum' in mode:
            self.crop_and_format_events = self.satsum
        else:
            raise ValueError("Unknown formatting method!")

    def plainsum(self, events_d, frame_ts, previous_ts):
        # todo input for 2d net, documentation
        new_events, _, _ = self.tf_crop_events_of_interest(events_d, frame_ts,
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
        new_events, _, _ = self.tf_crop_events_of_interest(events_d, frame_ts,
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
