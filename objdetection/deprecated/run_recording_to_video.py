import json
import os
import socket
import sys

import cv2
import numpy as np
import skvideo.io
import tensorflow as tf

sys.path.append(os.getcwd()[:os.getcwd().index('objdetection')])
from objdetection.rgb2events.nets import network_factory
from utils_visualisation import fun_general as vis_util

UDP_IP = "127.0.0.1"
UDP_PORT = 6785
BUFFERSIZE = 43200
MODEL_SAVE_PATH = "/home/ale/datasets/zuriscapes/Logs_official/Logs/" \
                  "03Dec2017_235135tfTRAINgaus40/10_gaus_model.ckpt"
NETWORK_MODEL = "ssd_davisSAE_master"
# Load mapping from integer class ID to sign name string
PATH2LABELS = "/home/ale/git_cloned/DynamicVisionTracking/objdetection/" \
              "SSDneuromorphic/labels/zauronscapes_label_map.json"
VIDEO_fold = "/home/ale/Videos/"
VIDEO_name = "output_long.mp4"
VIDEO_fullpath = VIDEO_fold + VIDEO_name
WIDTH = 240
HEIGHT = 180
CHANNELS = 3
with open(PATH2LABELS, "r") as f:
    raw_dict = json.load(f)
    # reformatting with key as int
    LABELS = {int(k): v for k, v in raw_dict.items()}


# ========================
def main():
    # returns a bool (True/False). If frame is read correctly, it will be
    # True. So you can
    # check end of the video by checking this return value.
    sock = socket.socket(socket.AF_INET,  # Internet
                         socket.SOCK_DGRAM)  # UDP
    sock.bind((UDP_IP, UDP_PORT))
    writer = skvideo.io.FFmpegWriter(
            VIDEO_fullpath, inputdict={'-r': '10'}, outputdict={'-r': '10'})
    try:
        # Launch the graph
        with tf.Graph().as_default(), tf.Session() as sess:
            # "Instantiate" neural network, get relevant tensors
            ssd_net = network_factory.get_network(NETWORK_MODEL)(
                    conf_thresh_cutoff=0.55)
            # Load trained model
            saver = tf.train.Saver()
            print('Restoring previously trained model at %s' % MODEL_SAVE_PATH)
            saver.restore(sess, MODEL_SAVE_PATH)
            # graph = tf.get_default_graph()
            while True:
                data, addr = sock.recvfrom(BUFFERSIZE)
                frame_udp = np.fromstring(data, dtype=np.uint8)
                frame_udp = np.reshape(frame_udp.reshape(180, 240),
                                       [1, 180, 240, 1])
                # scale between -1 and 1
                frame_udp = np.divide(frame_udp, 255 / 2) - 1
                # pass through the network
                events_out, classes_out, boxes_out, scores_out = \
                    sess.run([ssd_net.fast_events, ssd_net.fast_classes,
                              ssd_net.fast_boxes, ssd_net.fast_scores
                              ],
                             feed_dict={ssd_net.events_in:   frame_udp,
                                        ssd_net.is_training: False
                                        })
                # Add rectangles
                vis_util.visualize_boxes_and_labels_on_image_array(
                        events_out, boxes_out, classes_out,
                        scores_out, LABELS,
                        min_score_thresh=ssd_net.conf_thresh_cutoff,
                        use_normalized_coordinates=True,
                        line_thickness=1)
                # Display the resulting frame
                cv2.imshow('video', events_out)
                writer.writeFrame(events_out)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit `q` to exit
                    break
    finally:
        cv2.destroyAllWindows()
        writer.close()
        sock.close()
        print("Socket closed!")


if __name__ == "__main__":
    main()
