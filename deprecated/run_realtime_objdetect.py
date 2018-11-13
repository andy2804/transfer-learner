"""
author: az

This script performs object detection over an incoming stream of events-images
connecting via udp to the java transmitter that belongs to the retina repository
"""
import json
import socket

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from objdetection.rgb2events.nets import network_factory
from utils_visualisation import fun_general as vis_util
from objdetection.meta.utils_generic.magic_constants import Davis240c

UDP_IP = "127.0.0.1"
UDP_PORT = 6785
BUFFERSIZE = 43200
MODEL_SAVE_PATH = "/home/ale/encoder/zuriscapes/Logs/Log_09Nov2017_083429" \
                  "/200_exp_model.ckpt"
# Load mapping from integer class ID to sign name string
PATH2LABELS = "/home/ale/git_cloned/DynamicVisionTracking/objdetection/" \
              "SSDneuromorphic/labels/zauron_label_map.json"
with open(PATH2LABELS, "r") as f:
    raw_dict = json.load(f)
    # reformatting with key as int
    LABELS = {int(k): v for k, v in raw_dict.items()}


# ========================
def main():
    camera = Davis240c()
    # set up connection
    sock = socket.socket(socket.AF_INET,  # Internet
                         socket.SOCK_DGRAM)  # UDP
    sock.bind((UDP_IP, UDP_PORT))
    try:
        # Launch the graph
        with tf.Graph().as_default(), tf.Session() as sess:
            fig = plt.figure()
            im = plt.imshow(np.zeros([camera.sizeH, camera.sizeW, 3],
                                     dtype=np.uint8), vmin=0, vmax=255)
            plt.show(block=False)
            # "Instantiate" neural network, get relevant tensors
            ssd_net = network_factory.get_network("ssd_davisSAE_master")()
            # Load trained model
            saver = tf.train.Saver()
            print('Restoring previously trained model at %s' % MODEL_SAVE_PATH)
            saver.restore(sess, MODEL_SAVE_PATH)
            # graph = tf.get_default_graph()
            while True:
                data, addr = sock.recvfrom(
                        BUFFERSIZE)  # buffer size is 1024 bytes
                frame_udp = np.fromstring(data, dtype=np.uint8)
                frame_udp = np.reshape(
                        frame_udp.reshape(camera.sizeH, camera.sizeW),
                        [1, camera.sizeH, camera.sizeW, 1])
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
                # add rectangles
                vis_util.visualize_boxes_and_labels_on_image_array(
                        events_out, boxes_out, classes_out,
                        scores_out, LABELS,
                        min_score_thresh=ssd_net.net_par.conf_thresh_cutoff,
                        use_normalized_coordinates=True,
                        line_thickness=1)
                # display the resulting frame
                im.set_data(events_out)
                fig.canvas.draw()
    finally:
        sock.close()
        print("Socket closed!")


if __name__ == "__main__":
    main()
