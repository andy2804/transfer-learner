"""
author: aa
"""
import cv2
import numpy as np


def undistort_image_cv2(img, cam_info):
    return cv2.undistort(img, cam_info.K, cam_info.D)


def undistort_image_cv2_full(img, cam_info):
    # h, w = size
    # Get New Camera Matrix with alpha = 1 (including black borders)
    new_cam_matrix, roi = cv2.getOptimalNewCameraMatrix(
            cam_info.K, cam_info.D, (cam_info.height, cam_info.width), alpha=1, )
    new_img = cv2.undistort(img, cam_info.K, cam_info.D, dst=None, newCameraMatrix=new_cam_matrix)

    return new_img, new_cam_matrix


def undistort_image_manual(img, cam_info, size):
    h, w = size

    map_x = np.ndarray(shape=(h, w, 1), dtype='float32')
    map_y = np.ndarray(shape=(h, w, 1), dtype='float32')

    map_x, map_y = cv2.initUndistortRectifyMap(
            cam_info.K, cam_info.D, cam_info.R, cam_info.K, (cam_info.width, cam_info.height),
            cv2.CV_32FC1, map_x, map_y)
    img_rectified = np.empty_like(img)

    return cv2.remap(img, map_x, map_y, cv2.INTER_CUBIC, img_rectified)


def undistort_image_manual_full(img, cam_info, size):
    h, w = size

    map_x = np.ndarray(shape=(h, w, 1), dtype='float32')
    map_y = np.ndarray(shape=(h, w, 1), dtype='float32')

    new_cam_matrix, roi = cv2.getOptimalNewCameraMatrix(
            cam_info.K, cam_info.D, (cam_info.height, cam_info.width), 1, newImgSize=(h, w))
    map_x, map_y = cv2.initUndistortRectifyMap(
            cam_info.K, cam_info.D, cam_info.R, new_cam_matrix, (w, h), cv2.CV_32FC1, map_x, map_y)
    img_rectified = np.empty_like(img)

    return cv2.remap(img, map_x, map_y, cv2.INTER_CUBIC, img_rectified)
