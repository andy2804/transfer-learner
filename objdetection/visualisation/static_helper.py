"""
A set of functions that are used for visualization.
These functions often receive an image, perform some visualization on the image.
The functions do not return a value, instead they modify the image itself.
"""

import collections

import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import cv2
import numpy as np
import six
import tensorflow as tf

_TITLE_LEFT_MARGIN = 10  # fixme what are these for?
_TITLE_TOP_MARGIN = 10  # fixme what are these for?

RED = (255, 0, 0)
BLUE = (0, 0, 255)
STANDARD_COLORS = [
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'SkyBlue', 'MistyRose',
    'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'MintCream', 'HotPink', 'SpringGreen', 'YellowGreen', 'LightYellow',
    'LightGreen', 'LightPink',
    'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue',
    'LimeGreen', 'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue',
    'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'SteelBlue', 'GreenYellow', 'Teal', 'Thistle', 'Tomato',
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'MediumOrchid',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'Linen', 'Magenta', 'MediumAquaMarine',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'SteelBlue', 'GreenYellow', 'Teal', 'Thistle', 'Tomato',
    'Turquoise', 'Violet', 'Wheat', 'White', 'WhiteSmoke', 'Yellow',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet'
]


def save_image_array_as_png(image, output_path):
    """ Saves an image (represented as a numpy array) to PNG.
    :param image: a numpy array with shape [height, width, 3]
    :param output_path: path to which image should be written
    :return
    """
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    with tf.gfile.Open(output_path, 'w') as fid:
        image_pil.save(fid, 'PNG')


def encode_image_array_as_png_str(image):
    """ Encodes a numpy array into a PNG string.
    :param image: a numpy array with shape [height, width, 3].
    :returns :PNG encoded image string.
    """
    image_pil = Image.fromarray(np.uint8(image))
    output = six.BytesIO()
    image_pil.save(output, format='PNG')
    png_string = output.getvalue()
    output.close()
    return png_string


def draw_keypoints_on_image_array(image,
                                  keypoints,
                                  color='red',
                                  radius=2,
                                  use_normalized_coordinates=True):
    """ Draws keypoints on an image (numpy array).
    :param image: a numpy array with shape [height, width, 3].
    :param keypoints: a numpy array with shape [num_keypoints, 2].
    :param color: color to draw the keypoints with. Default is red.
    :param radius: keypoint radius. Default value is 2.
    :param use_normalized_coordinates: if True (default), treat keypoint 
    values as relative to the image.  Otherwise treat them as absolute.
  """
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    draw_keypoints_on_image(image_pil, keypoints, color, radius,
                            use_normalized_coordinates)
    np.copyto(image, np.array(image_pil))


def draw_keypoints_on_image(image,
                            keypoints,
                            color='red',
                            radius=2,
                            use_normalized_coordinates=True):
    """ Draws keypoints on an image.
    :param image: a PIL.Image object.
    :param keypoints: a numpy array with shape [num_keypoints, 2].
    :param color: color to draw the keypoints with. Default is red.
    :param radius: keypoint radius. Default value is 2.
    :param use_normalized_coordinates: if True (default), treat keypoint 
    values as relative to the image.  Otherwise treat them as absolute.
  """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    keypoints_x = [k[1] for k in keypoints]
    keypoints_y = [k[0] for k in keypoints]
    if use_normalized_coordinates:
        keypoints_x = tuple([im_width * x for x in keypoints_x])
        keypoints_y = tuple([im_height * y for y in keypoints_y])
    for keypoint_x, keypoint_y in zip(keypoints_x, keypoints_y):
        draw.ellipse([(keypoint_x - radius, keypoint_y - radius),
                      (keypoint_x + radius, keypoint_y + radius)],
                     outline=color, fill=color)


def draw_mask_on_image_array(image, mask, color='red', alpha=0.7):
    """ Draws mask on an image.
    :param image: uint8 numpy array with shape (img_height, img_height, 3)
    :param mask: a float numpy array of shape (img_height, img_height) with
                values between 0 and 1
    :param color: color to draw the keypoints with. Default is red.
    :param alpha: transparency value between 0 and 1. (default: 0.7)
    :raises ValueError: On incorrect data type for image or masks.
    """
    if image.dtype != np.uint8:
        raise ValueError('`image` not of type np.uint8')
    if mask.dtype != np.float32:
        raise ValueError('`mask` not of type np.float32')
    if np.any(np.logical_or(mask > 1.0, mask < 0.0)):
        raise ValueError('`mask` elements should be in [0, 1]')
    rgb = ImageColor.getrgb(color)
    pil_image = Image.fromarray(image)

    solid_color = np.expand_dims(
            np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
    pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
    pil_mask = Image.fromarray(np.uint8(255.0 * alpha * mask)).convert('L')
    pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
    np.copyto(image, np.array(pil_image.convert('RGB')))


def build_vertical_verbose_overview(
        snapshots, titles, scale=1, lines=None, cvt_color=True, bgr2rgb=True):
    """
    Takes list of tuples of images in snapshots and first stacks the tuple in a horizontal,
    then stacks the result in a vertical way to build a verbose overview.
    Can also take titles and adds them in the same manner to the corresponding pictures as
    defined in snapshots.
    If defined images are scaled to specified value.
    If defined, lines are drawn over the images
    :param snapshots: list of tuple of np.arrays --> [(img1_left, img1_right)]
    :param titles: list of tuple of strings --> [(title1_left, title1_right)]
    :param scale: float
    :param lines: int number of lines to be drawn, None to not draw any lines
    :param cvt_color: bool
    :param bgr2rgb: bool
    :return: np.array of stacked images
    """
    stack = []
    for i in range(len(snapshots)):
        if lines is not None:
            stack.append(visualize_stereo_images(snapshots[i], titles=titles[i], scale=scale,
                                                 n_lines=lines[i], cvt_color=cvt_color,
                                                 bgr2rgb=bgr2rgb))
        else:
            stack.append(visualize_stereo_images(snapshots[i], titles=titles[i], scale=scale,
                                                 cvt_color=cvt_color, bgr2rgb=bgr2rgb))
    overview = np.vstack(stack)
    cv2.imshow('FOV Fitting Overview', overview)
    cv2.waitKey(33)


def crop_and_resize_to_shape(img_input, target_shape, camera_info=None):
    """
    Combination of fov_fitters crop and resize for live zauron output
    :param img_input:
    :param target_shape:
    :param camera_info:
    :return:
    """
    img = np.copy(img_input)
    aspect_ratio = img.shape[1] / img.shape[0]
    target_aspect_ratio = target_shape[1] / target_shape[0]

    if aspect_ratio > target_aspect_ratio:
        new_width = int(img.shape[0] * target_aspect_ratio)
        delta_width = int((img.shape[1] - new_width) / 2)
        img_cropped = img[0:img.shape[0], delta_width:img.shape[1] - delta_width]
        if camera_info is not None:
            camera_info.K[0, 2] = camera_info.K[0, 2] * img_cropped.shape[1] / img.shape[1]
            camera_info.width = new_width
    elif aspect_ratio < target_aspect_ratio:
        new_height = int(img.shape[1] / target_aspect_ratio)
        delta_height = int((img.shape[0] - new_height) / 2)
        img_cropped = img[delta_height:img.shape[0] - delta_height, 0:img.shape[1]]
        if camera_info is not None:
            camera_info.K[1, 2] = camera_info.K[1, 2] * img_cropped.shape[0] / img.shape[0]
            camera_info.height = new_height
    else:
        img_cropped = img

    if camera_info is not None:
        scale_factor_h = target_shape[0] / img_input.shape[0]
        scale_factor_w = target_shape[1] / img_input.shape[1]
        camera_info.K[0, 0] = camera_info.K[0, 0] * scale_factor_w
        camera_info.K[0, 2] = camera_info.K[0, 2] * scale_factor_w
        camera_info.K[1, 1] = camera_info.K[1, 1] * scale_factor_h
        camera_info.K[1, 2] = camera_info.K[1, 2] * scale_factor_h
        camera_info.height, camera_info.width = target_shape[0], target_shape[1]
    return cv2.resize(img_cropped, target_shape[1::-1]), camera_info


def create_overlay_from_images(img_1, img_2, bgr2rgb=False):
    """
    Takes two images and overlays img_2 onto img_1 with alpha of 0.5.
    If necessary, reshapes the second to the size of the fist one.
    :param img_1: np.array of shape [height, width, 3]
    :param img_2: np.array of shape [height, width, 3]
    :param bgr2rgb: bool, if to convert bgr to rgb images
    :return: np.array with overlaid images
    """
    overlay = np.copy(img_2)
    img_to_overlay = np.copy(img_1)
    if bgr2rgb:
        img_to_overlay = cv2.cvtColor(img_to_overlay, cv2.COLOR_BGR2RGB)
    alpha = 0.5
    if img_1.shape[0] != img_2.shape[0] or img_1.shape[1] != img_2.shape[1]:
        img_to_overlay = cv2.resize(img_to_overlay, dsize=overlay.shape[:2])
    cv2.addWeighted(overlay, alpha, img_to_overlay, 1 - alpha, 0, dst=img_to_overlay)
    return img_to_overlay


def visualize_stereo_images(imgs_input, n_lines=0, scale=1, titles=('', ''), cvt_color=True,
                            bgr2rgb=False):
    """
    Takes two images and stacks them together in a horizontal way.
    Takes parameter cvt_color and bgr2rgb if needed, so grayscale images are converted to
    color ones and bgr images are converted to rgb ones.
    If necessary will add text overlay titles at the bottom if specified
    If lines are specified, it will draw n amount of lines over the image (for rectification)
    If scale is specified, images are resized to scale
    :param imgs_input: np.array of shape [height, width] or [height, width, 3] --> set cvt_color!
    :param n_lines: int amount of lines to be draw, 0 = off
    :param scale: float scaling amount
    :param titles: tuple of strings, Title for left and right image
    :param cvt_color: bool, if to convert grayscale images to color images
    :param bgr2rgb: bool, if to convert bgr input to rgb
    :return: stacked np.array
    """
    imgs = [np.copy(imgs_input[0]), np.copy(imgs_input[1])]

    # Color conversions
    if bgr2rgb:
        imgs[1] = cv2.cvtColor(imgs[1], cv2.COLOR_BGR2RGB)

    # Stack images and draw lines if necessary
    width = imgs[0].shape[1]
    height = imgs[0].shape[0]
    for idx, img in enumerate(imgs):
        img = cv2.resize(img, (int(width * scale), int(height * scale)))
        if cvt_color:
            if len(img.shape) <= 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if n_lines != 0:
            for line in range(0, img.shape[0], (int(img.shape[0] / n_lines))):
                img[line, :] = (0, 224, 0)
        if titles[idx] != '':
            img = add_text_overlay(img, titles[idx])
        imgs[idx] = img

    return np.hstack((imgs[0], imgs[1]))


def visualize_points(imgs_input, points, scale=1):
    """
    Draws circles around given points
    :param imgs_input: takes two corresponding images as np.array of shape [height, width, 3]
    :param points: array of points. point[0] = x, point[1] = y
    :param scale: float scale image to desired size
    :return: image with points
    """
    imgs = [np.copy(imgs_input[0]), np.copy(imgs_input[1])]
    width = imgs[0].shape[1]
    height = imgs[0].shape[0]
    for i in range(2):
        imgs[i] = cv2.resize(imgs[i], (int(width * scale), int(height * scale)))
        if len(imgs[i].shape) <= 2:
            imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_GRAY2RGB)
        for p in points[i]:
            px = int(p[0][0] * scale)
            py = int(p[0][1] * scale)
            cv2.circle(imgs[i], (px, py), 4, (0, 255, 0))
            cv2.imshow('SIFT Matches', visualize_stereo_images(imgs, titles=(
                'SIFT Matches DVS', 'SIFT Matches RGB'), bgr2rgb=True))
            cv2.waitKey(33)


def visualize_rgb_detections(img, obj_detected, labels=None, agnostic_mode=False):
    """ Visualization of the results of a detection.
    :param agnostic_mode:
    :param labels:
    :param img:
    :param obj_detected:
    :type obj_detected: ObjectDetected
    :return:
    """
    img_copy = np.copy(img)
    visualize_boxes_and_labels_on_image_array(
            img_copy,
            obj_detected.boxes,
            obj_detected.classes.astype(np.int32),
            obj_detected.scores,
            labels,
            use_normalized_coordinates=True,
            line_thickness=1,
            agnostic_mode=agnostic_mode)
    return img_copy


def draw_img_from_events(events, shape):
    """
    Draws a black / white image representation of events
    :param events: event_array
    :param shape: shape of the output image [height, width]
    :return: normalized image
    """
    img_events = np.zeros(shape[:2], dtype=np.uint8)
    for e in events:
        y, x = e[2:0:-1].astype(np.int)
        img_events[y, x] = 1
    # todo check if it's not better to normalize by number of events
    return cv2.normalize(img_events, img_events, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)


def draw_overlay_from_events(events_input, img_input, time_window=33, max_images=10):
    """
    Draw events on images according to time window
    :param max_images: int max images to return
    :param events_input: event_array
    :param img_input: np.array of shape [height, width, 3]
    :param time_window: int time window in [ms]
    :return: list of images
    """
    # Create a copy of input
    img = np.copy(img_input)
    events = np.copy(events_input)
    if len(img.shape) <= 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if len(events) == 0:
        return None

    # Define event time windows
    ts_start = events[0, 0]
    ts_end = events[-1, 0]
    if (ts_end - ts_start) <= 1 * 1e9:  # should be less than 1 second
        time_bins = np.arange(ts_start, ts_end, int(time_window * 1e6), dtype=np.int64)
        events[:, 0] = np.digitize(events[:, 0], time_bins, right=True)

        frames = []
        for i in range(1, min(len(time_bins) + 1, max_images) + 1):
            frame = np.copy(img)
            # add events corresponding to i
            for event in events[events[:, 0] == i]:
                frame[event[2], event[1], :] = RED if event[3] == 0 else BLUE
            frames.append(frame)
        if max_images == 1:
            return frames[-1]
        return frames
    return None


def add_text_overlay(img_in, title, overlay=True, fontsize=None):
    """
    Adds an text overlay at the bottom of the image.
    If overlay=True --> the text will be added overlaid on the image.
    If overlay=False --> expand image and add title below the image
    :param img_in: np.array of shape [height, width, 3]
    :param title: string
    :param overlay: bool
    :return: np.array with text overlaid on image
    """
    from PIL import Image
    from PIL import ImageFont
    from PIL import ImageDraw

    if fontsize is None:
        fontsize = int(max(min(20, img_in.shape[0] / 10), 8))
    pad = int(fontsize / 4)
    font = ImageFont.truetype(
            "/usr/share/fonts/truetype/ubuntu-font-family/UbuntuMono-R.ttf", fontsize)
    text_w, text_h = font.getsize(title)

    img = np.copy(img_in)
    if not overlay:
        img = cv2.copyMakeBorder(img_in, 0, text_h + 2 * pad, 0, 0, cv2.BORDER_CONSTANT, value=0)

    img_h, img_w = img.shape[:2]
    x_pos, y_pos = (img_w / 2 - (text_w / 2), img_h - text_h - pad)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    draw.rectangle((x_pos - pad, y_pos - pad, x_pos + text_w + pad, y_pos + text_h + pad),
                   fill='black')
    draw.text((x_pos, y_pos), title, fill='white', font=font)
    return np.array(img)


def scale_to_box(img_box, shape):
    """
    Takes an image of arbitrary size and aspect ratio and streches according to its
    aspect ratio to a given new shape
    :param img_box: np.array of shape [height, width, 3]
    :param shape: list of [height, width]
    :return: np.array with streched image
    """
    # Create empty black image
    target_height = shape[0]
    target_width = shape[1]
    img_output = np.zeros((target_height, target_width, 3), np.uint8)
    ratio = img_box.shape[1] / img_box.shape[0]
    target_ratio = target_width / target_height

    # scale until boundaries of box
    if ratio > target_ratio:
        img_box_cp = cv2.resize(img_box, (target_width, int(target_width / ratio)))
    elif ratio < target_ratio:
        img_box_cp = cv2.resize(img_box, (int(target_height * ratio), target_height))
    elif ratio == target_ratio:
        img_box_cp = cv2.resize(img_box, (target_width, target_height))

    # copy box crops into new image
    x_start = int((img_output.shape[1] - img_box_cp.shape[1]) / 2)
    y_start = int((img_output.shape[0] - img_box_cp.shape[0]) / 2)
    if len(img_box_cp.shape) < 3:
        img_box_cp = cv2.cvtColor(img_box_cp, cv2.COLOR_GRAY2RGB)
    img_output[y_start:y_start + img_box_cp.shape[0], x_start:x_start + img_box_cp.shape[1],
    :] = img_box_cp
    return img_output


def draw_bounding_box_on_image_array(image, ymin, xmin, ymax, xmax,
                                     color='red', thickness=4,
                                     display_str_list=(),
                                     use_normalized_coordinates=True, difficult=False):
    """ Adds a bounding box to an image (numpy array).
    :param image: a numpy array with shape [height, width, 3].
    :param ymin: ymin of bounding box in normalized coordinates (same below).
    :param xmin: xmin of bounding box.
    :param ymax: ymax of bounding box.
    :param xmax: xmax of bounding box.
    :param color: color to draw bounding box. Default is red.
    :param thickness: line thickness. Default value is 4.
    :param display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    :param use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
    """
    # todo fix class visualisation
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color,
                               thickness, display_str_list,
                               use_normalized_coordinates, difficult)
    np.copyto(image, np.array(image_pil))


def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax,
                               color='red', thickness=4, display_str_list=(),
                               use_normalized_coordinates=True, difficult=False):
    """Adds a bounding box to an image.
  Each string in display_str_list is displayed on a separate line above the
  bounding box in black text on a rectangle filled with the input 'color'.

  :param image: a PIL.Image object.
  :param ymin: ymin of bounding box.
  :param xmin: xmin of bounding box.
  :param ymax: ymax of bounding box.
  :param xmax: xmax of bounding box.
  :param color: color to draw bounding box. Default is red.
  :param thickness: line thickness. Default value is 4.
  :param display_str_list: list of strings to display in box
                      (each to be shown on its own line).
  :param use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
    """
    draw = ImageDraw.Draw(image, 'RGBA')
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=thickness, fill=color)
    try:
        # AA (apr '18): made font size invariant to image size (--> DPI)
        font = ImageFont.truetype('arial.ttf', max(int(im_height / 100 * 10), 32))  # 8% of imgsize
    except IOError:
        font = ImageFont.load_default()

    text_bottom = top
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        diff_width, diff_height = font.getsize('!')
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
                [(left, text_bottom - text_height - 2 * margin),
                 (left + text_width + 2 * margin, text_bottom)],
                fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                  display_str, fill='black', font=font)

        # Draw exclamation mark for difficult to detect objects
        if difficult:
            draw.rectangle(
                    [(left - diff_width - 2 * margin - 2, text_bottom - text_height - 2 * margin),
                     (left - 2, text_bottom)],
                    fill='orange', outline='black')
            draw.text((left - diff_width - margin, text_bottom - text_height - margin),
                      '!', fill='black', font=font)

        text_bottom -= text_height - 2 * margin


def draw_bounding_boxes_on_image_array(image, boxes,
                                       color='red', thickness=4,
                                       display_str_list_list=(), difficult=None):
    """Draws bounding boxes on image (numpy array).
    :param image: a numpy array object.
    :param boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
           The coordinates are in normalized format between [0, 1].
    :param color: color to draw bounding box. Default is red.
    :param thickness: line thickness. Default value is 4.
    :param display_str_list_list: list of list of strings.
            a list of strings for each bounding box.
            The reason to pass a list of strings for a
            bounding box is that it might contain
            multiple labels.

    :raise ValueError: if boxes is not a [N, 4] array
  """
    image_pil = Image.fromarray(image)
    draw_bounding_boxes_on_image(image_pil, boxes, color, thickness,
                                 display_str_list_list, difficult)
    np.copyto(image, np.array(image_pil))


def draw_bounding_boxes_on_image(image,
                                 boxes,
                                 color='red',
                                 thickness=4,
                                 display_str_list_list=(),
                                 difficult=None):
    """Draws bounding boxes on image.
    :param image: a PIL.Image object.
    :param boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax,xmax).
           The coordinates are in normalized format between [0, 1].
    :param color: color to draw bounding box. Default is red.
    :param thickness: line thickness. Default value is 4.
    :param display_str_list_list: list of list of strings.
                           a list of strings for each bounding box.
                           The reason to pass a list of strings for a
                           bounding box is that it might contain
                           multiple labels.

    :raises ValueError: if boxes is not a [N, 4] array
  """
    boxes_shape = boxes.shape
    if not boxes_shape:
        return
    if len(boxes_shape) != 2 or boxes_shape[1] != 4:
        raise ValueError('Input must be of size [N, 4]')
    if difficult is None:
        difficult = [0] * boxes_shape[0]
    for i in range(boxes_shape[0]):
        display_str_list = ()
        if display_str_list_list:
            display_str_list = display_str_list_list[i]
        draw_bounding_box_on_image(image, boxes[i, 0], boxes[i, 1], boxes[i, 2],
                                   boxes[i, 3], color, thickness,
                                   display_str_list, difficult[i])


def visualize_boxes_and_labels_on_image_array(image, boxes, classes, scores, labels,
                                              instance_masks=None, keypoints=None,
                                              use_normalized_coordinates=False,
                                              max_boxes_to_draw=25, min_score_thresh=.5,
                                              agnostic_mode=False, line_thickness=4,
                                              difficult=None, alpha=None):
    """ Overlay labeled boxes on an image with formatted scores and label names.
    This function groups boxes that correspond to the same location
    and creates a display string for each detection and overlays these
    on the image.  Note that this function modifies the image array in-place
    and does not return anything.

    :param image: uint8 numpy array with shape (img_height, img_width, 3)
    :param boxes: a numpy array of shape [N, 4]
    :param classes: a numpy array of shape [N]
    :param scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes with no scores.
    :param labels: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    :param instance_masks: a numpy array of shape [N, image_height, image_width], can
      be None
    :param keypoints: a numpy array of shape [N, num_keypoints, 2], can be None
    :param use_normalized_coordinates: whether boxes is to be interpreted as
      normalized coordinates or not.
    :param max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw all boxes.
    :param min_score_thresh: minimum score threshold for a box to be visualized
    :param agnostic_mode: boolean (default: False) controlling whether to evaluate in
      class-agnostic mode or not.  This mode will display scores but ignore classes.
    :param line_thickness: integer (default: 4) controlling line width of the boxes.
    """

    # Create a display string (and color) for every box location, group any
    # boxes that correspond to the same location.
    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)
    box_to_instance_masks_map = {}
    box_to_keypoints_map = collections.defaultdict(list)
    box_to_difficult_map = collections.defaultdict(int)

    # Draw all boxes onto image.
    if difficult is None or len(difficult) == 0:
        difficult = [0] * len(boxes)
    assert len(difficult) == len(boxes)

    if not max_boxes_to_draw:
        max_boxes_to_draw = len(boxes)
    for i in range(min(max_boxes_to_draw, len(boxes))):
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            if instance_masks is not None:
                box_to_instance_masks_map[box] = instance_masks[i]
            if keypoints is not None:
                box_to_keypoints_map[box].extend(keypoints[i])
            else:
                if not agnostic_mode:
                    if classes[i] in labels.keys():
                        class_name = labels[classes[i]]["name"]
                    else:
                        class_name = 'N/A'
                    if scores is None:
                        display_str = '{}'.format(class_name)
                    else:
                        display_str = '{}: {}%'.format(class_name, int(100 * scores[i]))
                else:
                    display_str = 'score: {}%'.format(int(100 * scores[i]))
                box_to_display_str_map[box].append(display_str)
                if agnostic_mode:
                    box_to_color_map[box] = 'DarkOrange'
                else:
                    box_to_color_map[box] = STANDARD_COLORS[
                        classes[i] % len(STANDARD_COLORS)]
                if alpha is not None:
                    box_to_color_map[box] = ImageColor.getrgb(box_to_color_map[box]) + (
                        int(alpha[i] * 255),)
                box_to_difficult_map[box] = difficult[i]

    for box, color in box_to_color_map.items():
        ymin, xmin, ymax, xmax = box
        if instance_masks is not None:
            draw_mask_on_image_array(
                    image,
                    box_to_instance_masks_map[box],
                    color=color)
        draw_bounding_box_on_image_array(
                image, ymin, xmin, ymax, xmax,
                color=color,
                thickness=line_thickness,
                display_str_list=box_to_display_str_map[box],
                use_normalized_coordinates=use_normalized_coordinates,
                difficult=box_to_difficult_map[box])
        if keypoints is not None:
            draw_keypoints_on_image_array(
                    image,
                    box_to_keypoints_map[box],
                    color=color,
                    radius=line_thickness,
                    use_normalized_coordinates=use_normalized_coordinates)
