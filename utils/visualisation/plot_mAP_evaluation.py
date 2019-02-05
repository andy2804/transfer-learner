"""
author: aa
"""

import numpy as np
from matplotlib import patches as mpatches
from matplotlib import pyplot as plt

COLORS = ['orange', 'blue']
CB_COLOR_PREFIX = 'dark'   # Prefix for confidence bound line color e.g. 'dark' or 'light'
ALPHA = 0.8
SMALL = 10
MEDIUM = 12
BIG = 18
TEXT_FAMILY = 'sans-serif'
TEXT_WEIGHT = 'medium'


def _set_fonts():
    """
    Set font family and sizes
    :return:
    """
    plt.rc('text', usetex=True)
    font = {'family': 'serif',
            'size':   MEDIUM
            }
    plt.rc('font', **font)
    plt.rc('axes', titlesize=MEDIUM)
    plt.rc('axes', labelsize=MEDIUM)
    plt.rc('xtick', labelsize=SMALL)
    plt.rc('ytick', labelsize=SMALL)
    plt.rc('figure', titlesize=BIG)


def plot_performance_metrics(corestats,
                             ap,
                             mAP,
                             labels,
                             filename='dataset',
                             relative_bar_chart=True,
                             label_filter=None,
                             plots_per_row=3,
                             plot_bars=True,
                             micro_averaged=True):
    """
    Create subplots for each class with accuracy,
    recall and amount of gt, fp, tp
    If specified, renders the bars of FP and TP relative to GT
    If more than one corestat is passed in the list, it will plot them over each other
    Corestats need to be tested on the same evaluation set and have the same amount of classes
    :param corestats: list of corestats, one with best mAP first
    :param ap: list of ap's
    :param labels: list of labels
    :param filename: name of plot to be plotted in the title and for exporting
    :param relative_bar_chart: plot bars for threshold 0.50
    :param label_filter:
    :param plots_per_row: if not plot bars, then # of plots per row
    :param plot_bars: plot bars for threshold 0.50
    :return:
    """
    _set_fonts()
    num_classes = len(corestats[0][0]['acc'])
    if label_filter is None:
        num_labels = num_classes
        label_filter = labels
    else:
        num_labels = len(label_filter)

    # Plot class-wise precision-recall curves
    if plot_bars:
        num_rows = num_labels
        fig, axs = plt.subplots(num_rows, 2, sharex='col', figsize=(8, num_labels * 3),
                                dpi=150, )  # gridspec_kw={'top':.9})
    else:
        num_rows = int(np.ceil(num_labels / plots_per_row))
        fig, axs = plt.subplots(num_rows, plots_per_row, figsize=(num_rows * 4, num_rows * 3))
    titlename = filename.replace('&', '\&').replace('_', '\_')
    fig.suptitle('Performance Metrics\n{}'.format(titlename))

    # Initialize values for each corestat
    thresholds = []
    mid_thresh = []
    max_score = []
    networks = [network.replace('_', '\_') for network in filename.split(' & ')]
    for key in corestats:
        thresholds.append(sorted(key.keys()))
        mid_thresh.append(thresholds[-1][int(len(thresholds[-1]) / 2)])
        max_score.append(np.max([_get_classes_max(num_classes, key, mid_thresh[-1], 'tp'),
                                 _get_classes_max(num_classes, key, mid_thresh[-1], 'fp'),
                                 _get_classes_max(num_classes, key, mid_thresh[-1], 'n_gt')]))

    # If not to plot bars, plot several precision-recall curves in a row to save space
    if plot_bars:
        plots_per_row = 1
        axs[0][1].set_title('Samples for thresh = %.2f' % mid_thresh[0])
    curr_cls = 1
    fig.tight_layout(rect=[0, 0.03, 1, 0.85 + 0.02 * num_rows])
    for curr_ax in range(num_rows):
        ax = axs[curr_ax]

        # Plot all curves in a row first
        for plot_id in list(range(plots_per_row)):
            ax[plot_id].set_title('Recall-Precision')
            if curr_cls in label_filter:
                # Subplots for recall-accuracy curves
                sorted_ap = [val[curr_cls] for val in ap]
                keys = np.argsort(sorted_ap[::-1])
                for idx, key in enumerate(keys):
                    _plot_acc_rec(ax[plot_id], corestats[key], ap[key], curr_cls, idx,
                                  thresholds[key], COLORS[key], mid_thresh[0])
                    ax[plot_id].set_ylabel(labels[curr_cls]['name'])

                # # RSS Paper single subplot export
                # extent = ax[plot_id].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                #
                # # Pad the saved area by 10% in the x-direction and 20% in the y-direction
                # fig.savefig('%s.pdf' % labels[curr_cls]['name'], bbox_inches=extent.expanded(1.3, 1.2))

                # Subplots for tp, fp and gt scores
                if plot_bars:
                    for idx, key in enumerate(keys):
                        _plot_bar_chart(ax[1], corestats[key], curr_cls, mid_thresh[key],
                                        max_score[key],
                                        COLORS[key], relative_bar_chart)
                    ax[1].text(0.98, 0.92, 'n_{%s} = %d' % (
                        labels[curr_cls]['name'], corestats[0][mid_thresh[0]]['n_gt'][curr_cls]),
                               size=SMALL, family=TEXT_FAMILY, weight=TEXT_WEIGHT,
                               color=COLORS[key],
                               horizontalalignment='right',
                               verticalalignment='center', transform=ax[1].transAxes)
            else:
                ax[plot_id].set_visible(False)
            curr_cls += 1

    # Prepare plot handles for legend
    plot_handles = []
    for idx in range(len(networks)):
        plot_handles.append(mpatches.Patch(color=COLORS[idx]))
    plt.figlegend(plot_handles, networks, loc='lower center', ncol=2, fontsize='small',
                  markerscale=4, shadow=False, frameon=False, fancybox=False)

    if micro_averaged:
        # Plot micro-averaged precision-recall curve
        fig2 = plt.figure(figsize=(4.5, 3), dpi=150)
        #fig2.suptitle('Micro-Averaged Performance Metrics')
        keys = np.argsort(mAP[::-1])
        ax = plt.gca()
        for idx, key in enumerate(keys):
            _plot_micro_acc_rec(ax, corestats[key], mAP[key], mid_thresh[key], COLORS[key], idx)
        fig2.tight_layout(rect=[0, 0.06, 1, 1])
        plt.figlegend(plot_handles, networks, loc='lower center', ncol=2, fontsize='small',
                      markerscale=4, shadow=False, frameon=False, fancybox=False)
    else:
        # Plot macro-averaged precision-recall curve
        # TODO to implement
        fig2 = plt.figure(figsize=(4.5, 3), dpi=150)
        #fig2.suptitle('Micro-Averaged Performance Metrics')
        keys = np.argsort(mAP[::-1])
        ax = plt.gca()
        for idx, key in enumerate(keys):
            _plot_micro_acc_rec(ax, corestats[key], mAP[key], mid_thresh[key], COLORS[key], idx)
        fig2.tight_layout(rect=[0, 0.06, 1, 1])
        plt.figlegend(plot_handles, networks, loc='lower center', ncol=2, fontsize='small',
                      markerscale=4, shadow=False, frameon=False, fancybox=False)
    return fig, fig2


def _plot_micro_acc_rec(ax, corestats, mAP, mid_thresh, color, idx, conf_level=.95,
                        conf_method="wilson"):
    """
    #todo
    :param ax:
    :param corestats:
    :param mAP:
    :param color:
    :param conf_level:
    :param conf_method:
    :return:
    """
    micro_acc, micro_rec = [], []
    _acc_ci, _rec_ci = [], []
    micro_acc_low, micro_acc_high, micro_rec_low, micro_rec_high = [], [], [], []
    for thresh in sorted(corestats.keys())[::-1]:
        micro_acc.append(corestats[thresh]['micro_acc'].estimate)
        micro_rec.append(corestats[thresh]['micro_rec'].estimate)

        # Get values for confidence bounds
        if conf_level is not None:
            try:
                _acc_ci.append(corestats[thresh]['micro_acc'].get_confidence_interval(
                        conf_level, conf_method))
                _rec_ci.append(corestats[thresh]['micro_rec'].get_confidence_interval(
                        conf_level, conf_method))
                micro_acc_low.append(_acc_ci[-1][0])
                micro_acc_high.append(_acc_ci[-1][1])
                micro_rec_low.append(_rec_ci[-1][0])
                micro_rec_high.append(_rec_ci[-1][1])
            except ValueError:
                print("Failed to compute accuracy intervals, skipping plots")
                micro_acc_low = []

    # Get position of mid_thresh
    micro_acc_mid = corestats[mid_thresh]['micro_acc'].estimate
    micro_rec_mid = corestats[mid_thresh]['micro_rec'].estimate

    # Actual plotting commands
    ax.stackplot(micro_rec, micro_acc, color=color, alpha=ALPHA, zorder=2)
    ax.plot(micro_rec, micro_acc, color=color, lw=1.5, zorder=3)
    ax.text(0.97, 0.93 - idx * 0.07, 'mAP: %.2f' % mAP, size=SMALL,
            family=TEXT_FAMILY, weight=TEXT_WEIGHT, color=color,
            horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)

    # If confidence values are present also plot them
    if micro_acc_low:
        ax.plot(micro_rec_low, micro_acc_low, '--', color=CB_COLOR_PREFIX + color, lw=1.5, zorder=3)
        ax.plot(micro_rec_high, micro_acc_high, '--', color=CB_COLOR_PREFIX + color, lw=1.5, zorder=3)
    ax.plot(micro_rec_mid, micro_acc_mid, color=CB_COLOR_PREFIX + color, marker='x', ms=6, mew=1.5,
            fillstyle='none',
            zorder=10)

    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')


def _plot_acc_rec(ax, corestats, ap, cls, idx, thresholds, color, mid_thresh, conf_level=.95,
                  conf_method="wilson"):
    """
    #todo
    Confidence interval computed as displacement from nominal accuracy taking into account both
    95% margin on recall value and on accuracy value.
    :param ax:
    :param corestats:
    :param cls:
    :param thresholds:
    :param color:
    :param mid_thresh:
    :return:
    """
    acc, rec = [], []
    _acc_ci, _rec_ci = [], []
    acc_low, acc_high, rec_low, rec_high = [], [], [], []
    # thresholds = np.array(thresholds[::-1])

    # Read out accuracy recall values for each treshold
    for thresh in thresholds[::-1]:
        acc.append(corestats[thresh]['acc'][cls].estimate)
        rec.append(corestats[thresh]['rec'][cls].estimate)
        if conf_level is not None:
            try:
                _acc_ci.append(corestats[thresh]['acc'][cls].get_confidence_interval(
                        conf_level, conf_method))
                _rec_ci.append(corestats[thresh]['rec'][cls].get_confidence_interval(
                        conf_level, conf_method))
                acc_low.append(_acc_ci[-1][0])
                acc_high.append(_acc_ci[-1][1])
                rec_low.append(_rec_ci[-1][0])
                rec_high.append(_rec_ci[-1][1])
            except ValueError:
                print("Failed to compute accuracy intervals, skipping plots")
                acc_low = []
    acc_mid = corestats[mid_thresh]['acc'][cls].estimate
    rec_mid = corestats[mid_thresh]['rec'][cls].estimate

    # Actual plotting commands
    ax.stackplot(rec, acc, color=color, alpha=ALPHA, zorder=2)
    ax.plot(rec, acc, '--', color=color, lw=1.5, zorder=3)

    # if the conf bounds have been filled
    if acc_low:
        # todo can be improved with filling the areas but plt.fill_between() requires same x
        ax.plot(rec_low, acc_low, '--', color=CB_COLOR_PREFIX + color, lw=1.5, zorder=3)
        ax.plot(rec_high, acc_high, '--', color=CB_COLOR_PREFIX + color, lw=1.5, zorder=3)
    ax.plot(rec_mid, acc_mid, color=CB_COLOR_PREFIX + color, marker='x', ms=6, mew=1.5, fillstyle='none',
            zorder=10)

    # Plot AP per class value
    ax.text(0.95, 0.90 - idx * 0.075, 'AP: %.2f' % (ap[cls]), size=SMALL,
            family=TEXT_FAMILY, weight=TEXT_WEIGHT, color=color,
            horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)


def _plot_bar_chart(ax, corestats, cls, mid_thresh, max_score, color, relative_bar_chart):
    """
    #todo
    :param ax:
    :param corestats:
    :param cls:
    :param mid_thresh:
    :param max_score:
    :param color:
    :param relative_bar_chart:
    :return:
    """
    tp = corestats[mid_thresh]['tp'][cls]
    fp = corestats[mid_thresh]['fp'][cls]
    n_gt = corestats[mid_thresh]['n_gt'][cls]

    # Calculate relative values
    if relative_bar_chart:
        tp = tp / n_gt
        fp = fp / n_gt
        n_gt = 1.0
        ax.set_xlim(0.0, 1.0)
    else:
        ax.set_xlim(0, max_score)
    scores = [tp, fp, n_gt]
    positions = np.arange(3) + 1

    # Acutal plotting command
    ax.barh(positions, scores, 0.5, align='center', color=color, edgecolor='black', lw=1,
            alpha=ALPHA)

    # Plot labels and adjust axes
    bar_labels = ['Tp', 'Fp', 'GT']
    ax.set_yticks(positions)
    ax.set_yticklabels(bar_labels)
    ax.set_ylim(0, 4)
    ax.spines['top'].set_color(None)
    ax.spines['right'].set_color(None)


def _get_classes_max(num_classes, corestats, thresh, stat_type):
    """
    Gets the maximum score of one stat type over all classes
    :param thresh:
    :param stat_type:
    :return:
    """
    max = 0
    for cls in range(1, num_classes + 1):
        score = corestats[thresh][stat_type][cls]
        max = score if score > max else max
    return max


def _get_classes_sum(corestats, threshold, stat_type):
    return sum([corestats[threshold][stat_type][i]] for i in
               corestats[threshold][stat_type])
