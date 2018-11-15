"""
author: aa
"""

import os
import pickle
import sys
import time
from datetime import timedelta

PROJECT_ROOT = os.getcwd()[:os.getcwd().index('transferlearning')]
sys.path.append(PROJECT_ROOT)

from transferlearning.filter.stats import TLStatistician
from utils.sheets_interface import GoogleSheetsInterface
from utils.static_helper import load_labels

INPUT_DIR = '/home/andya/external_ssd/wormhole_learning/dataset/testing'
OUTPUT_DIR = '/home/andya/external_ssd/wormhole_learning/dataset/testing'
TESTNAME = ['ZAURON_DAYONLY_ROI_CLIPPED']
LABELS = 'zauron_label_map.json'
WORKSHEET = 'zauron_dataset'

RECREATE_ALL_PLOTS = True
UPLOAD_TO_SHEETS = True

"""
#
# USAGE:
# --------------------------------------------------------------
# Specify the input and output folders containing the pickled corestats data.
# Specify the testname, which corresponds to the filename of the pickled data,
# excluding the "_corestats.pickle" ending.
# It will automatically load the file and plot the data.
# --------------------------------------------------------------
# If more than one file is specified in TESTNAME (as a list), then these plots
# will be plotted over each other for comparison purposes
# --------------------------------------------------------------
# - If RECREATE_ALL_PLOTS is set to true, all pickled data is loaded and all plots are updated.
# - If UPLOAD_TO_SHEETS is set to true, the result will be re-uploaded to Google Sheets. Only
#   tick, if initial upload from evaluator failed.
#
"""


def load_stats_from_files(files):
    return [pickle.load(open(file, 'rb')) for file in files]


def upload_results(sheets, stats, testname, labels):
    # Prepare data for upload to google sheets result page
    values = []
    for idx in range(len(labels)):
        number = len(stats.get_tlscores(label_filt=idx + 1, tl_keep_filt=1))
        diff = len(stats.get_tlscores(label_filt=idx + 1, tl_keep_filt=0))
        values.append('%d (%d)' % (number, diff))
    values.append(len(stats.get_tlscores()))

    try:
        sheets.upload_data(WORKSHEET, 'B', 'I', testname, values)
    except:
        print("\nUpload Error!")


def plot_from_stats(corestats, files):
    labels = load_labels(LABELS)
    stats = TLStatistician(tl_score_threshold=0, labels_file=LABELS)
    sheets = GoogleSheetsInterface()
    for idx in range(len(corestats)):
        testname = os.path.basename(files[idx]).strip("_objstats.pickle")
        print("Evaluating %s" % testname)
        stats.load(corestats[idx])
        stats.make_plots(save_plots=True, output_dir=OUTPUT_DIR, filename=testname, show_plots=True)
        if UPLOAD_TO_SHEETS:
            upload_results(sheets, stats, testname, labels)


def main():
    """
    Just plots the data from previous evaluation runs
    Please specifiy just the testname and it will automatically load
    corresponding corestats and AP files
    :return:
    """
    t0 = time.time()

    # Get File paths
    files = []
    if RECREATE_ALL_PLOTS:
        files = [os.path.join(INPUT_DIR, file) for file in os.listdir(INPUT_DIR) if
                 '.pickle' in file]
    else:
        for file in TESTNAME:
            files.append(os.path.join(INPUT_DIR, file + '_objstats.pickle'))

    # Load all corresponding corestats
    corestats = load_stats_from_files(files)

    plot_from_stats(corestats, files, )
    deltatime = timedelta(seconds=time.time() - t0)
    print("\nPlotting completed in:\t", deltatime)


if __name__ == '__main__':
    main()
