"""
author:az
"""
import argparse
import os
import sys

PROJECT_ROOT = os.getcwd()[:os.getcwd().index('operating_domain')]
sys.path.append(PROJECT_ROOT)
from objdetection.deprecated.operating_domain.day_vs_night import SvmDayNight


def _parse_input():
    parser = argparse.ArgumentParser(
            description='Split dataset into day/night images')
    parser.add_argument('--src_dir', type=str,
                        default="/home/ale/datasets/day_vs_night/val2017",
                        help='Output directory containing images at day time')
    parser.add_argument('--day_dir', type=str,
                        default="day",
                        help='Output directory containing images at day time')
    parser.add_argument('--night_dir', type=str,
                        default="night",
                        help='Output directory containing images at night time')
    _args = parser.parse_args()
    return _args


def _load_images_path(src_dir):
    return [os.path.join(src_dir, f) for f in os.listdir(src_dir) if
            os.path.splitext(f)[1] in [".jpg", ".png"]]


def main(args):
    classifier = SvmDayNight()
    classifier.load_model()
    image_paths = _load_images_path(args.src_dir)
    classifier.split_dataset(image_paths=image_paths, day_out=args.day_dir,
                             night_out=args.night_dir)


if __name__ == '__main__':
    args = _parse_input()
    main(args)
    print("Dataset segmentation completed")
