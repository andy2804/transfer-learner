"""
author: az
"""
import argparse
import os
import sys

PROJECT_ROOT = os.getcwd()[:os.getcwd().index('operating_domain')]
sys.path.append(PROJECT_ROOT)
from objdetection.deprecated.operating_domain.day_vs_night import SvmDayNight


def _parse_input():
    parser = argparse.ArgumentParser(
            description='Train svm for day/night classification of images')
    parser.add_argument('--day_dir', type=str,
                        default="/home/ale/encoder/day_vs_night/day_images",
                        help='Directory containing images at day time')
    parser.add_argument('--night_dir', type=str,
                        default="/home/ale/encoder/day_vs_night/night_images",
                        help='Directory containing images at night time')
    _args = parser.parse_args()
    return _args


def main(_args):
    """
    Main function. Trains a classifier
    """
    classifier = SvmDayNight()
    classifier.train(
            training_path_day=_args.day_dir, training_path_night=_args.night_dir)
    classifier.save_model()
    while True:
        try:
            image_url = input("Input an image url (enter to exit): ")
            # image_url = 'https://thumbs.dreamstime.com/b/asian-traffic-scene-night-4574194.jpg'
            if not image_url:
                break
            try:
                img, features = classifier.process_image_url(image_url)
                print(classifier.predict([features, ]))
            except OSError:
                pass
        except (KeyboardInterrupt, EOFError):
            break


if __name__ == '__main__':
    args = _parse_input()
    main(args)
