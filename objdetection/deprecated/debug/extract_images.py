import argparse
import sys
from os import makedirs
from os.path import isfile

import cv2
from cv_bridge import CvBridge
from rosbag import Bag


def convertToCV(imgMsg):
    bridge = CvBridge()
    return bridge.compressed_imgmsg_to_cv2(imgMsg, "passthrough")


# initialize parser
parser = argparse.ArgumentParser(description='Extract Images form rosbag')
parser.add_argument('input', type=str)
parser.add_argument('topic', type=str)
parser.add_argument('--output', type=str, default='output', required=False)

# read arguments
args = parser.parse_args()
rosbagIn = args.input
desTopic = args.topic
output = args.output

# check bag
if not isfile(rosbagIn):
    print('Input error')
    exit(1)

try:
    makedirs(output)
except:
    pass

# process bag
print('Input Rosbag file: %s' % rosbagIn)
print('Topic to extract:  %s' % desTopic)
bag = Bag(rosbagIn)
topics = bag.get_type_and_topic_info().topics

counter = 0
if desTopic in topics:
    for _, msg, _ in bag.read_messages(desTopic):
        cvImg = convertToCV(msg)
        cv2.imwrite('%s/img_%04i.jpg' % (output, counter), cvImg)
        print('\rWrote img_%04i.jpg' % counter)
        sys.stdout.flush()
        counter += 1

bag.close()
