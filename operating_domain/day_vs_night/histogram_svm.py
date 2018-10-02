"""
author: az
"""
import os
import pickle
import random
from functools import partial
from io import BytesIO
from multiprocessing.pool import Pool

import numpy as np
import requests
from PIL import Image
from sklearn import metrics
from sklearn import model_selection
from sklearn.svm import SVC


class SvmDayNight:
    def __init__(self):
        self.best_estimator = None
        self.label = {'night': 1, 'day': 0}

    def _process_directory(self, directory, max_files=200):
        """Returns an array of feature vectors for all the image files in a
        directory (and all its subdirectories). Symbolic links are ignored.
        Args:
          directory (str): directory to process.
        Returns:
          list of list of float: a list of feature vectors.
        """
        if not os.path.isdir(directory):
            raise IOError('{} is not a directory'.format(directory))
        training = []
        files = os.listdir(directory)
        if len(files) > max_files:
            files = random.sample(files, max_files)
        for i, file_name in enumerate(files):
            print("\rExtracting features from image {:d} of {:d}".format(i, len(files)),
                  end='', flush=True)
            file_path = os.path.join(directory, file_name)
            _, img_feature = self.process_image_file(file_path)
            if img_feature is not None:
                training.append(img_feature)
        return training

    def process_image_file(self, file_path):
        im = Image.open(file_path)
        return im, self._process_image(im)

    def process_image_url(self, image_url):
        """Given an image URL it returns its feature vector
        Args:
          image_url (str): url of the image to process.
        Returns:
          list of float: feature vector.
        Raises:
          Any exception raised by urllib2 requests.
          IOError: if the URL does not point to a valid file.
        """

        r = requests.get(image_url, stream=True)
        try:
            img = Image.open(BytesIO(r.content))
            return img, self._process_image(img)
        except OSError:
            print("Image readout did not work out, please try again")
            raise

    def _process_image(self, im, bins=5):
        """Given a PIL Image object it returns its feature vector.
        :type im: PIL.Image.
        :type bins: int. Number of block to subdivide the RGB space into.
        :returns np.array:
          list of float: feature vector if successful. None if the image is not
          RGB.
        """
        if not im.mode == 'RGB':
            print("Image is not RGB, will be skipped.")
            return
        return self._extract_features(im, bins)

    @staticmethod
    def _extract_features(im, bins=5):
        feature = np.zeros((bins ** 3))
        im_np = np.array(im)
        pixel_count = im_np.size
        im_np = np.floor_divide(im_np, (256 / bins)).astype(np.int)
        rgb_weights = np.array([1, bins, bins ** 2])
        im_np *= rgb_weights
        idxes = np.sum(im_np, axis=-1)
        for idx in idxes.flat:
            feature[idx] += 1
        return (feature / pixel_count).tolist()

    def train(self, training_path_day, training_path_night, print_metrics=True):
        """Trains a classifier. training_path_a and training_path_b should be
        directory paths and each of them should not be a subdirectory of the other
        one. training_path_a and training_path_b are processed by
        process_directory().
        Args:
          training_path_day (str): directory containing sample images of class A.
          training_path_night (str): directory containing sample images of class B.
          print_metrics  (boolean, optional): if True, print statistics about
            classifier performance.
        Returns:
          A classifier (sklearn.svm.SVC).
        """
        print("Initializing training...")
        training_day = self._process_directory(training_path_day)
        training_night = self._process_directory(training_path_night)
        # data contains all the training data (a list of feature vectors)
        data = training_day + training_night
        # target is the list of target classes for each feature vector: a '1' for
        # class A and '0' for class B
        target = [self.label['day']] * len(training_day) + [self.label['night']] * len(
                training_night)
        # Train test split
        x_train, x_test, y_train, y_test = model_selection.train_test_split(
                data, target, test_size=0.20)
        # Parameter search space
        parameters = {'kernel': ['linear', 'rbf'],
                      'C':      [1, 10, 100, 1000],
                      'gamma':  [0.1, 0.01, 0.001, 0.0001],
                      }
        # search for the best classifier within the search space and return it
        clf = model_selection.GridSearchCV(SVC(), parameters).fit(x_train, y_train)
        self.best_estimator = clf.best_estimator_
        if print_metrics:
            print('\nParameters: {}'.format(clf.best_params_))
            print('\nBest classifier score')
            print(metrics.classification_report(y_test, self.best_estimator.predict(x_test)))
            print("Confusion matrix: ",
                  metrics.confusion_matrix(y_test, self.best_estimator.predict(x_test)))
        return

    def predict(self, features):
        if self.best_estimator is None:
            raise RuntimeError('A classifier need to be trained or loaded before prediction')
        else:
            label_id = self.best_estimator.predict(features)
            return list(self.label.keys())[list(self.label.values()).index(label_id)]

    def save_model(self, filename='best_svm.pickle'):
        # todo pickle self.best_estimator
        if self.best_estimator is not None:
            try:
                pickle.dump(self.best_estimator, open(filename, 'wb'))
                print("Saved classifier model as \"{}\"".format(filename))
            except IOError:
                print("An error occurred trying to pickle the model.")
        else:
            print("Missing classifier to pickle. Need to run 'train' function before?")

    def load_model(self, filename='best_svm.pickle'):
        try:
            self.best_estimator = pickle.load(open(filename, 'rb'))
        except IOError:
            print("File to load not found")

    def split_dataset(self, image_paths, day_out, night_out):
        try:
            os.mkdir(os.path.join(os.path.dirname(image_paths[0]), day_out))
            os.mkdir(os.path.join(os.path.dirname(image_paths[0]), night_out))
        except FileExistsError:
            print("Folder day and/or night already exist")

        splitdata_partial = partial(
                self._split_dataset, day_out=day_out, night_out=night_out)
        with Pool() as p:
            p.map(splitdata_partial, image_paths, chunksize=30)

    def _split_dataset(self, image, day_out, night_out):
        img, features = self.process_image_file(image)
        if features is not None:
            label = self.predict([features, ])
            # move file
            old_dir = os.path.dirname(image)
            if label == 'day':
                new_dir = os.path.join(old_dir, day_out)
            elif label == 'night':
                new_dir = os.path.join(old_dir, night_out)
            else:
                raise ValueError('Predicted label has not been recognized')
            # actual move
            os.rename(image, os.path.join(new_dir, os.path.basename(image)))
