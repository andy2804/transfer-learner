from objdetection.meta.utils_events.events_transform import EventsTransform


class ThreeD(EventsTransform):
    def __init__(self, time_window=0.05):
        super().__init__(time_window)

    # todo input for network(this is only for plotting)
    def crop_and_format_events(self, events_d, frame_ts, previous_ts,
                               time_bins):
        _, idx_pos, idx_neg = self.tf_crop_events_of_interest(events_d, frame_ts,
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
