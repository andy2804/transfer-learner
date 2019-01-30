"""
author: aa & az

The class LearningFilter contains the core functionality to analyze the
observability/representation of the same object in different domains.
At the moment it's limited to event-based and integrative cameras (i.e. standard frame-based
cameras).
Using the `_compute_score` method we compute an empirical score used to infer whether or not the
same object is visible in both domains.
#todo
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils.static_helper import load_labels
from utils.visualisation.static_helper import add_text_overlay, draw_bounding_box_on_image_array, \
    scale_to_box, visualize_stereo_images


class MultiModalObserver:
    def __init__(self):
        self._labels = load_labels('zauron_label_map.json')

        # Only initialize in the instance that really needs it
        self._img_blender = None

    def _create_verbose_image(self, score, keep, class_id, box, hand_label,
                              img_from_box,
                              img_to_box,
                              img_from_labeled,
                              img_to_labeled,
                              img_from_gradient,
                              img_to_gradient):
        """
        Method to create a stacked verbose overview of images,
        that we use as input to apply transfer learning to.
        Creates a 3x2 stacked output image.
        :param score: learning filter score
        :param keep: if we are keeping the label
        :param class_id: detected class id
        :param box: absolute bounding box values
        :param hand_label: if we are hand-labeling learning filter output
        :param img_from_box: takes in 3 channel rgb np.array
        :param img_to_box: takes in 3 channel rgb np.array
        :param img_from_labeled: takes in 3 channel rgb np.array
        :param img_to_labeled: takes in 3 channel rgb np.array
        :param img_from_gradient: takes in 1 channel gray-scale gradient np.array
        :param img_to_gradient: takes in 1 channel gray-scale gradient np.array
        :return:
        """
        # Debug image stacked
        if img_to_labeled is None:
            img_to_labeled = img_from_labeled

        # Create stacked input image
        img_input_stack = visualize_stereo_images((img_from_labeled, img_to_labeled),
                                                  titles=('TL From Image', 'TF To Image'),
                                                  scale=2, cvt_color=False)
        img_from_gradient_bbox = cv2.cvtColor(img_from_gradient, cv2.COLOR_GRAY2RGB)
        img_to_gradient_bbox = cv2.cvtColor(img_to_gradient, cv2.COLOR_GRAY2RGB)
        draw_bounding_box_on_image_array(img_from_gradient_bbox, box[0], box[1], box[2], box[3],
                                         use_normalized_coordinates=False, thickness=1,
                                         color='blue')
        draw_bounding_box_on_image_array(img_to_gradient_bbox, box[0], box[1], box[2], box[3],
                                         use_normalized_coordinates=False, thickness=1,
                                         color='blue')

        # Create gradient stack
        img_gradient_stack = visualize_stereo_images((img_from_gradient_bbox, img_to_gradient_bbox),
                                                     titles=('Gradient Image', 'Events visualized'),
                                                     scale=2, cvt_color=True)
        img_from_box_scaled = scale_to_box(img_from_box, img_from_labeled.shape[0:2])
        img_to_box_scaled = scale_to_box(img_to_box, img_from_labeled.shape[0:2])
        if keep and self._img_blender is not None:
            self._img_blender._update_avg_img(img_to_box_scaled, class_id)

        # Create scaled to box stack
        img_scaled_stack = visualize_stereo_images((img_from_box_scaled, img_to_box_scaled),
                                                   scale=2,
                                                   cvt_color=False)
        img_scaled_stack = add_text_overlay(img_scaled_stack, (
                '%s matching score: %i%% --> %s' % (
            self._labels[class_id]['name'], score, str(keep).capitalize())), overlay=True)
        overview = np.vstack((img_input_stack, img_gradient_stack, img_scaled_stack))
        cv2.imshow('Input Images & Gradients', overview)
        if hand_label:
            return cv2.waitKey()
        else:
            cv2.waitKey(33)
            return None

    @staticmethod
    def _compute_observability_score(input_crops, type='rgb', verbose=''):
        """
        Call for observability computation depending on input type
        :param input_crops: Takes in a tuple of images cropped to the detected object
        :param type:
        :return:
        """
        if type == 'rgb':
            return MultiModalObserver._compute_overlap(input_crops, 'rgb', verbose)
        elif type == 'events' or 'events_np':
            return MultiModalObserver._compute_overlap(input_crops, 'events', verbose)
        else:
            print('Undefined type was specified!')

    @staticmethod
    def _compute_overlap(input_crops, mode='events', verbose=''):
        """ Observability of events inferred from rgb frames
        Calculates score how well event frame overlaps with rgb frame
        to determine the activity in the event frame
        :param input_crops[0]:
        :param input_crops[1]:
        :return: score
        """
        # Activity score
        # activity_score = 1 + (abs(1 - (np.sum(input_crops[1]) / np.sum(input_crops[0]))) * -1)

        # Overlap score
        img_box = np.empty_like(input_crops[0], dtype=np.float64)
        cv2.normalize(input_crops[0], img_box, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
        img_events_box = np.empty_like(input_crops[1], dtype=np.float64)
        img_events_box_blur = cv2.GaussianBlur(input_crops[1], (3, 3), 0)
        cv2.normalize(img_events_box_blur, img_events_box,
                      norm_type=cv2.NORM_MINMAX,
                      dtype=cv2.CV_64F)
        if mode == 'rgb':
            img_sum = np.sum(img_box)
        elif mode == 'events':
            img_sum = np.sum(img_events_box)
        else:
            img_sum = np.sum(img_box)
        overlap_score = np.sum(np.sqrt(np.multiply(img_box, img_events_box))) / img_sum

        # Weighting
        # score = min(max((activity_score + 2 * overlap_score) / 3.0, 0.0), 1.0)
        score = int(np.nan_to_num(min(max(overlap_score, 0.0), 1.0)) * 100.0)

        # Verbose Image
        if verbose != '':
            img_box = scale_to_box((img_box * 255).astype(np.uint8), (200, 300))
            img_events_box = scale_to_box((img_events_box * 255).astype(np.uint8), (200, 300))
            img_box = add_text_overlay(img_box, "Main Sensor", overlay=False)
            img_events_box = add_text_overlay(img_events_box, "Aux Sensor", overlay=False)
            img_stack = np.hstack((img_box, img_events_box))
            img_stack = add_text_overlay(img_stack, "Score: %d" % score, overlay=True)

            if verbose == 'cv2':
                cv2.imshow('Learning Filter Score', img_stack)
                cv2.waitKey(1)
            elif verbose == 'plot':
                plt.figure("figure", figsize=(8, 4))
                plt.imshow(img_stack)
                plt.xticks([])
                plt.yticks([])
                plt.show()

        return score

    @staticmethod
    def _calc_rgb_entropy(input_crops, verbose=False):
        """
        Calculates the Shannon Entropy of an Image
        :param input_crops[1: rgb image or a cropped section
        :return: entropy score
        """
        # todo rename this method, decouple observability from entropy

        # Calc histogram
        hist = cv2.calcHist([input_crops[1]], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, None)
        entropy = 0

        # Calc Entropy
        for hist_val in hist:
            if hist_val > 0:
                # Shannon Entropy for discrete random variable
                # entropy += hist_val * np.log2(1/hist_val)

                # Shannon differential Entropy for continuous random variable
                entropy -= hist_val * np.log2(hist_val)

        # Verbose Image
        if verbose:
            fig, axs = plt.subplots(2, 1, figsize=(4, 6))
            img_scaled = scale_to_box(input_crops[1], (200, 300))
            axs[0].imshow(img_scaled)
            axs[0].get_xaxis().set_ticks([])
            axs[0].get_yaxis().set_ticks([])
            axs[1].bar(np.arange(0, 256), hist.squeeze(), facecolor='black')
            axs[1].text(0.95, 0.90, 'Entropy: %.2f' % entropy, size=10, weight='bold',
                        horizontalalignment='right', transform=plt.gca().transAxes)
            plt.show()

        return entropy

    @staticmethod
    def _box_norm_to_abs(box, img):
        """
        Convert normalized coordinates to image coordinates
        :param box: [y_min, x_min, y_max, x_max]
        :param img:
        :return:
        """
        return [int(x) for x in np.multiply(box, np.concatenate([img.shape[0:2]] * 2))]

    @staticmethod
    def _abs_box_to_norm(box, img):
        """
        Convert absolute positions back to normalized coordinates
        :param box:
        :param img:
        :return:
        """
        return np.divide(box, np.concatenate([img.shape] * 2))

    @staticmethod
    def _get_box_crop(img, box):
        """
        Crops to active region
        :param img:
        :param box: [y_min, x_min, y_max, x_max]
        :return:
        """
        img_box = img[box[0]:box[2], box[1]:box[3]]
        return img_box

    @staticmethod
    def _get_events_until_ts(events, to_ts, time_window):
        """ Number of events inside the box for a certain time window normalized by its perimeter
        :param events: [ts, x, y, pol]
        :param to_ts: [nsec]
        :param time_window: [ms]
        :return:
        """
        # Converting units
        time_window = int(time_window * 1e6)

        events_cropped = events[np.where(
                np.logical_and(to_ts - time_window < events[:, 0], events[:, 0] <= to_ts))]
        return events_cropped

    @staticmethod
    def _filter_img(img_in, filter='gradient', normalize=True, mode='rgb'):
        """
        Filters the given image with the filter specified
        and normalizes the image to 0 - 255
        :param img_in:
        :param filter:
        :param normalize:
        :return:
        """
        img = cv2.cvtColor(np.copy(img_in), code=cv2.COLOR_RGB2GRAY)
        if mode == 'events':
            img = MultiModalObserver._compute_absolute_events(img)
        elif mode == 'events_np' or mode == 'rgb':
            pass
        if filter == 'gradient':
            output = MultiModalObserver._compute_gradient(img)
        elif filter == 'laplacian':
            output = MultiModalObserver._compute_laplacian(img)
        elif filter == 'canny':
            output = MultiModalObserver._compute_canny(img)
        elif filter == 'grayscale':
            output = img
        else:
            raise ValueError("Filter method not available")
        if normalize:
            output = cv2.normalize(output, output,
                                   alpha=0,
                                   beta=255,
                                   norm_type=cv2.NORM_MINMAX,
                                   dtype=cv2.CV_8U)

        return output

    @staticmethod
    def _compute_gradient(img_in):
        """
        Computes the gradient of the input image
        :param img_in:
        :return:
        """
        dx = cv2.Sobel(img_in, cv2.CV_64FC1, 1, 0, ksize=3)
        dy = cv2.Sobel(img_in, cv2.CV_64FC1, 0, 1, ksize=3)
        return np.sqrt(dx ** 2 + dy ** 2)

    @staticmethod
    def _compute_canny(img_in, th_low=80, th_high=160):
        """
        Computes the canny edge image
        :param img_in:
        :param th_low:
        :param th_high:
        :return:
        """
        return cv2.Canny(img_in, th_low, th_high)

    @staticmethod
    def _compute_laplacian(img_in):
        """
        Computes the laplacian of the input image
        :param img_in:
        :return:
        """
        return cv2.Laplacian(cv2.GaussianBlur(img_in, (3, 3), 0),
                             cv2.CV_64FC1,
                             ksize=3,
                             scale=1,
                             delta=0)

    @staticmethod
    def _compute_absolute_events(img_in):
        """
        Shifts the grayscale events image mean by -127.5 and then takes all absolute values,
        which are normalized back to a range of 0-255.
        Turns image black and all location where events occur to white pixels.
        :param img_in:
        :return:
        """
        img = np.copy(img_in.astype(np.float64))
        return (np.abs((img - 127.5)) * 2).astype(np.uint8)
