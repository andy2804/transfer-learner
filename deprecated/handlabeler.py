import os
import sys

import cv2
import numpy as np

sys.path.append(os.getcwd()[:os.getcwd().index('objdetection')])
from utils_visualisation import static_helper


class Labeler:
    def __init__(self, labels_dict):
        self._newrefPt = [-1, -1]
        self._newrefID = []
        self._labels_dict = labels_dict
        self._drawing = False
        self._image = None
        self._clone = None
        self._boxes = None
        self._ids = None

    def _getnewbox(self):
        # format the box to p1=(ymin xmin) and p2=(ymax xmax)
        assert len(self._newrefPt) == 4
        new = [max(0, min(self._newrefPt[0::2])),
               max(0, min(self._newrefPt[1::2])),
               min(180, max(self._newrefPt[0::2])),
               min(240, max(self._newrefPt[1::2]))]
        return np.expand_dims(np.array(new), axis=0) / np.array(
                [180, 240, 180, 240])

    def _click_and_crop(self, event, x, y, flags, param):
        # grab references to the global variables
        # if the left mouse button was clicked, record the starting
        # (y, x) coordinates and indicate that cropping is being
        # performed
        if event == cv2.EVENT_LBUTTONDOWN:
            self._newrefPt = [y, x]
            self._drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and self._drawing:
            self._image = self._clone.copy()
            p1 = (self._newrefPt[1], self._newrefPt[0])
            cv2.rectangle(self._image, p1, (x, y), (0, 255, 0), 1)
        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (y, x) coordinates and input the class
            self._drawing = False
            self._newrefPt.extend([y, x])
            p1 = (self._newrefPt[1], self._newrefPt[0])
            cv2.rectangle(self._image, p1, (x, y), (0, 255, 0), 1)
            cv2.imshow("image", self._image)

    def interface_click_callback(self, image, boxes, ids):
        self._clone = image.copy()
        self._image = image
        self._boxes = np.array([], dtype=np.float32) if np.size(
                boxes) == 0 else boxes
        self._ids = np.array([], dtype=np.int64) if np.size(ids) == 0 else ids
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("image", 800, 600)
        cv2.setMouseCallback("image", self._click_and_crop)
        # keep looping until the 'q' key is pressed
        while True:
            # display the image and wait for a keypressc
            cv2.imshow("image", self._image)
            key = cv2.waitKey(1) & 0xFF
            # if 'q' key is pressed, break from the loop
            if key == ord("q"):
                break
            # 4 coordinates -> a rectangle has been selected
            if len(self._newrefPt) == 4:
                print(
                        "Insert the category number:\ntruck : 1\nbicycle : "
                        "2\nperson: 3\ntram/bus: 4\ncar : "
                        "5\nmotorcycle: 6")
                raw_input = cv2.waitKey(0) & 0xFF
                id_input = int(chr(raw_input))
                if id_input in range(1, len(self._labels_dict.keys()) + 1):
                    self._newrefID.append(id_input)
                    print("New id category:", self._newrefID)
                    new_box = self._getnewbox()
                    self._boxes = np.vstack([self._boxes, new_box]) if np.size(
                            self._boxes) > 0 else new_box
                    new_id = np.array(self._newrefID, dtype=np.int64)
                    self._ids = np.append(self._ids, new_id)
                    fun_general.visualize_boxes_and_labels_on_image_array(
                            self._image, self._boxes, self._ids,
                            None,
                            self._labels_dict,
                            use_normalized_coordinates=True,
                            line_thickness=1
                    )
                    cv2.imshow("image", self._image)
                    self._newrefPt = []
                    self._newrefID = []
                else:
                    raise ValueError("The input value is not a possible class!")
        # close all open windows
        cv2.destroyAllWindows()
        return self._boxes, self._ids
