"""
author: aa
"""
import os
import pickle
import sys
import time
from datetime import timedelta
from pprint import pprint

PROJECT_ROOT = os.getcwd()[:os.getcwd().index('objdetection')]
sys.path.append(PROJECT_ROOT)

from objdetection.detector.detector import ARCH_DICT
from objdetection.evaluator.evaluator import EvaluatorFrozenGraph
from utils.sheets_interface import GoogleSheetsInterface
from utils.static_helper import load_labels
from utils.visualisation.plot_mAP_evaluation import plot_performance_metrics

INPUT_DIR = '/home/andya/external_ssd/wormhole_learning/results'
OUTPUT_DIR = '/home/andya/external_ssd/wormhole_learning/results'
TESTNAME = ['ssd_inception_v2_kaist_dayonly_1_kaist_night_rgb',
            'ssd_inception_v2_kaist_ir035_rgb050_RGB_3_kaist_night']
LABELS = 'zauron_label_map.json'

# To plot all classes, set LABEL_FILTER = None
LABEL_FILTER = None

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


def print_stats(corestat, AP, mAP):
    print('\nCorestats:')
    pprint(corestat)
    print('AP per Class:')
    pprint(AP)
    print('mAP:\t%.2f' % mAP)


def save_plot(plot, testname):
    # Save the plot as pdf
    try:
        if plot is not None:
            plot.savefig('{}.pdf'.format(testname))
    except PermissionError:
        print("\n Permission error during saving plots!")


def load_corestats_from_files(files):
    return [pickle.load(open(file, 'rb')) for file in files]


def compute_scores_from_corestats(corestats):
    AP = []
    mAP = []
    for corestat in corestats:
        thresholds = sorted(corestat.keys())
        ap, map = EvaluatorFrozenGraph.compute_ap(corestat, thresholds)
        AP.append(ap)
        mAP.append(map)
    return AP, mAP


def upload_results(sheets, network, testset, AP, mAP):
    try:
        sheets.upload_evaluation(network, testset, AP, mAP)
    except Exception as e:
        print("\nUpload Error: %s" % e)


def split_testname(testname):
    network = 'N/A'
    testset = 'N/A'
    for key, value in ARCH_DICT.items():
        if value in testname:
            network = value
            testset = testname.replace(network + '_', '')
            break
    return (network, testset)


def plot_from_corestats(corestats, files, comparison_plot=False):
    sheets = GoogleSheetsInterface()
    labels = load_labels(LABELS)
    AP, mAP = compute_scores_from_corestats(corestats)
    if comparison_plot:
        testnames = [os.path.splitext(os.path.basename(file))[0] for file in files]
        networks = []
        for testname in testnames:
            network, _ = split_testname(testname)
            networks.append(network)
        testname = '\nvs\n'
        testname = testname.join(networks)
        plot = plot_performance_metrics(corestats, AP, labels, testname,
                                        relative_bar_chart=True, label_filter=LABEL_FILTER)
        for idx in range(len(corestats)):
            print_stats(corestats[idx], AP[idx], mAP[idx])
        pdf_file = os.path.join(OUTPUT_DIR, testname)
        save_plot(plot, pdf_file)
    else:
        for idx in range(len(corestats)):
            testname = os.path.splitext(os.path.basename(files[idx]))[0]
            network, testset = split_testname(testname)
            plot = plot_performance_metrics([corestats[idx]], [AP[idx]], labels, network,
                                            relative_bar_chart=True, label_filter=LABEL_FILTER)
            print_stats(corestats[idx], AP[idx], mAP[idx])
            pdf_file = os.path.join(OUTPUT_DIR, testname)
            save_plot(plot, pdf_file)
            if UPLOAD_TO_SHEETS:
                upload_results(sheets, network, testset, AP[idx], mAP[idx])


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
            files.append(os.path.join(INPUT_DIR, file + '.pickle'))

    # Load all corresponding corestats
    corestats = load_corestats_from_files(files)

    # Initialize GoogleSheetInterface

    if RECREATE_ALL_PLOTS:
        plot_from_corestats(corestats, files, comparison_plot=False)
    else:
        plot_from_corestats(corestats, files, comparison_plot=True)

    deltatime = timedelta(seconds=time.time() - t0)
    print("\nPlotting completed in:\t", deltatime)


if __name__ == '__main__':
    main()
