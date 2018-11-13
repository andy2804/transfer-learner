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
                        default="/media/sdc/andya/wormhole_learning"
                                "/zauron_recordings/"
                                "20181026/day_cloudy_2018-10-26-10-34-00.bag")
    parser.add_argument("--output_dir", help="Output directory.",
                        default="/home/azanardi/pictures/zauron_eye", required=False)
    parser.add_argument("--image_topic", help="Image topic.",
                        default="/zed/right/image_raw/compressed", required=False)
    parser.add_argument("--offset_frame", help="offset frame to begin with extraction",
                        default=2000)
    parser.add_argument("--max_frames", help="Max number of frames to extract",
                        default=1000)

    args = parser.parse_args()

    print("Extract images from {} on topic {} into {}".format(
            args.bag_file, args.image_topic, args.output_dir))

    bag = rosbag.Bag(args.bag_file, "r")
    bridge = CvBridge()
    count = 0
    for topic, msg, t in bag.read_messages(topics=[args.image_topic]):
        if count in range(args.offset_frame, args.offset_frame + args.max_frames):
            cv_img = bridge.compressed_imgmsg_to_cv2(msg, "passthrough")
            # cv_img = cv_img[..., ::-1]
            cv2.imwrite(os.path.join(args.output_dir, "img{:06d}.png".format(count)), cv_img)

            if count % 50 == 0:
                print("Wrote {:d} images".format(count - args.offset_frame))

        count += 1

    bag.close()
    return


if __name__ == '__main__':
    main()
