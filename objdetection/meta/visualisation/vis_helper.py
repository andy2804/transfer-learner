"""
author: az
"""
import os
from copy import copy

import cv2
import numpy as np
from moviepy.editor import ImageSequenceClip
from scipy.misc import imresize

from rosbag_converter.utils_rosbag_converter.static_helper import copy_events_from_chunk
from objdetection.meta.visualisation.static_helper import (create_overlay_from_images,
                                                           add_text_overlay)

RED = (255, 0, 0)
BLUE = (0, 0, 255)


def _maybe_out_dir(output_dir):
    if not os.path.isdir(output_dir):
        # Create missing folders
        try:
            os.makedirs(output_dir)
        except OSError:
            print("Failed to create output directory")


def _chunks(data, n_chunks):
    # todo what's this function for?
    """Yield successive n-sized chunks from l."""
    for i in range(0, data.shape[0], n_chunks):
        yield data[i:i + n_chunks]


def create_media_from_chunk(chunk,
                            events_topic='/dvs/events',
                            frame_topic='/pylon_rgb/image_raw',
                            offset_ts=0,
                            media_format='mp4',
                            media_resolution=10,
                            media_slowdown=10,
                            output_dir="~/",
                            media_name=None):
    """
    # todo Support to also add bounding boxes
    :param media_format:
    :param media_slowdown:
    :param events_topic:
    :param media_name:
    :param frame_topic:
    :param media_resolution: temporal resolution for the gif in ms
    :param chunk:
    :type chunk: list of DataInstances
    :param output_dir:
    :return:
    """
    _maybe_out_dir(output_dir)
    if media_name is None:
        media_name = 'chunk_%s.%s' % (str(chunk[0][frame_topic].ts.to_sec()), media_format)
    # frame_size = chunk[0][frame_topic].image.shape
    # start_ts = chunk[0][frame_topic].ts.to_nsec() - media_resolution / 2 * 1e6
    # end_ts = chunk[-1][frame_topic].ts.to_nsec() + media_resolution / 2 * 1e6
    # bins = np.arange(start_ts, end_ts, media_resolution * 1e6)
    frame_size = chunk[0][frame_topic].image.shape
    start_ts = int(chunk[0][frame_topic].ts.to_nsec() - int(media_resolution * 1e6) + 1)
    end_ts = int(chunk[-1][frame_topic].ts.to_nsec() + int(media_resolution * 1e6) + 1)
    bins = np.arange(start_ts, end_ts, int(media_resolution * 1e6), dtype=np.int64)
    bins_offset = np.arange(start_ts - offset_ts, end_ts - offset_ts, int(media_resolution * 1e6),
                            dtype=np.int64)

    # concat events and digitize ts
    all_events = copy_events_from_chunk(chunk, ts_interval=(start_ts, end_ts - offset_ts),
                                        events_topic=events_topic)
    all_events[:, 0] = np.digitize(all_events[:, 0], bins_offset, right=True)

    # concat images and digitize ts
    all_frames = [imresize(np.copy(datainstance[frame_topic].image), frame_size)
                  for datainstance in chunk]
    all_frames_ts = np.asarray(
            [copy(datainstance[frame_topic].ts.to_nsec()) for datainstance in chunk])
    all_frames_ts = np.digitize(all_frames_ts, bins, right=True)

    # create frames with overlapped events
    frames_list = []  #
    for i in range(1, len(bins) + 1):
        # add last frame as base
        idx_new_frame = [idx for idx, value in enumerate(all_frames_ts) if value <= i][-1]
        new_frame = copy(all_frames[idx_new_frame])
        # add events corresponding to i
        for e in all_events[all_events[:, 0] == i]:
            if len(new_frame.shape) <= 2:
                new_frame = cv2.cvtColor(new_frame, cv2.COLOR_GRAY2RGB)
            new_frame[e[2], e[1], :] = RED if e[3] == 0 else BLUE
        frames_list.append(new_frame)

    # generate gif
    fps = round(1 / (media_resolution * 1e-3 * media_slowdown))
    path_to_file = os.path.join(output_dir, media_name)
    compose_media_from_frames(frames_list, fps, path_to_file)


def compose_media_from_frames(frames, fps, file):
    clip = ImageSequenceClip(frames, fps=fps)
    format = os.path.splitext(file)[1]
    if 'gif' in format:
        clip.write_gif(file)
        print("\nGif stored at ", file)
    elif 'mp4' in format:
        clip.write_videofile(file)
        print("\nVideo stored at ", file)
    else:
        print("\nRequested format for the media is not available!")


def create_overlay_from_chunk(chunk, output_dir):
    """
    Create overlay images from specified chunk
    :param chunk:
    :param output_dir:
    :return:
    """
    # todo function to be adjusted or erased
    _maybe_out_dir(output_dir)
    frames = []

    for index, data_instance in enumerate(chunk):
        # DVS & RGB Images
        img_dvs = np.copy(data_instance['/dvs/image_raw'].image)
        img_rgb = np.copy(data_instance['/pylon_rgb/image_raw'].image)
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        events = np.copy(data_instance['/dvs/events'])
        height, width = img_dvs.shape[:2]

        # For event iteration
        num = 2  # number of event images (#TODO might have to be changed based on time bins)
        indices = np.arange(0, events.shape[0], step=1, dtype=int)
        len_chunk = events.shape[0] // num
        indices_gen = _chunks(indices, len_chunk)

        # Images Overlay
        overlay = create_overlay_from_images(img_rgb, img_dvs)

        # Event Overlay Image
        for event_idx in range(num):
            indices = next(indices_gen)
            event_imgs = np.zeros((height, width))
            event_imgs[events[indices, 2], events[indices, 1]] = 2 * (events[indices, 3]) - 1

            img_events_overlay = img_rgb.copy()
            if len(img_events_overlay.shape) <= 2:
                img_events_overlay = cv2.cvtColor(img_events_overlay, cv2.COLOR_GRAY2RGB)
            where_1 = np.where(event_imgs == 1)
            where_neg1 = np.where(event_imgs == -1)
            img_events_overlay[where_1[0], where_1[1], 0] = 255  # Positive event override pixels
            img_events_overlay[where_1[0], where_1[1], 1:] = 0
            img_events_overlay[where_neg1[0], where_neg1[1], 2] = 255
            img_events_overlay[where_neg1[0], where_neg1[1], :2] = 0

        img_dvs = cv2.resize(img_dvs, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
        img_rgb = cv2.resize(img_rgb, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
        img_dvsrgb_overlay = cv2.resize(img_dvsrgb_overlay, (width * 2, height * 2),
                                        interpolation=cv2.INTER_CUBIC)
        img_events_overlay = cv2.resize(img_events_overlay, (width * 2, height * 2),
                                        interpolation=cv2.INTER_CUBIC)
        img_dvs = add_text_overlay(img_dvs, 'DVS_Rectified')
        img_rgb = add_text_overlay(img_rgb, 'PYLON_Rectified')
        img_dvsrgb_overlay = add_text_overlay(img_dvsrgb_overlay, 'DVS_PYLON_Overlay')
        img_events_overlay = add_text_overlay(img_events_overlay, 'PYLON_Events_Overlay')

        new_frame = np.vstack((np.hstack((img_dvs, img_rgb)),
                               np.hstack((img_dvsrgb_overlay, img_events_overlay))))
        # cv2.imwrite(os.path.join(output_dir, 'stacked_images_%.4d.png' % index), new_frame)
        frames.append(new_frame)

    ts = chunk[0]['/pylon_rgb/image_raw'].ts.to_sec()
    file = os.path.join(output_dir, 'stacked_images_%s.mp4' % str(ts))
    compose_media_from_frames(frames, 10, file)
