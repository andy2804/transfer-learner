"""
Extract images from a rosbag.
"""

import argparse
import os

import cv2
import rosbag
from cv_bridge import CvBridge


def main():
    """Extract a folder of images from a rosbag.
    """
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("--bag_file", help="Input ROS bag.", required=False,
                        default="/media/ale/dubi_usb1/zauron_seye/eth_2018-10-26-10-06-30.bag")
    parser.add_argument("--output_dir", help="Output directory.",
                        default="/home/ale/Pictures/zauron_eye", required=False)
    parser.add_argument("--image_topic", help="Image topic.",
                        default="/zed/right/image_raw/compressed", required=False)

    args = parser.parse_args()

    print("Extract images from {} on topic {} into {}".format(
            args.bag_file, args.image_topic, args.output_dir))

    bag = rosbag.Bag(args.bag_file, "r")
    bridge = CvBridge()
    count = 0
    for topic, msg, t in bag.read_messages(topics=[args.image_topic]):

        cv_img = bridge.compressed_imgmsg_to_cv2(msg, "passthrough")
        # cv_img = cv_img[..., ::-1]
        cv2.imwrite(os.path.join(args.output_dir, "frame{:d}.png".format(count)), cv_img)

        if count % 50 == 0:
            print("Wrote {:d} images".format(count))

        count += 1

    bag.close()

    return


if __name__ == '__main__':
    main()
