"""
author: aa
"""

import numpy as np
from matplotlib import pyplot as plt

COLORS = ['blue', 'orange']
ALPHA = 0.8


def _set_fonts():
    """
    Set font family and sizes
    :return:
    """
    # plt.rc('text', usetex=True)
    SMALL = 9
    MEDIUM = 12
    BIG = 18
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
                             labels,
                             filename='dataset',
                             relative_bar_chart=True,
                             label_filter=None):
    """
    Create subplots for each class with accuracy,
    recall and amount of gt, fp, tp
    If specified, renders the bars of FP and TP relative to GT
    If more than one corestat is passed in the list, it will plot them over each other
    Corestats need to be tested on the same evaluation set and have the same amount of classes
    :param corestats: list of corestats, one with best mAP first
    :param ap: list of ap's
    :param labels:
    :param filename:
    :param relative_bar_chart:
    :param label_filter:
    :return:
    """
    _set_fonts()
    num_classes = len(corestats[0][0]['acc'])
    if label_filter is None:
        num_labels = num_classes
        label_filter = labels
    else:
        num_labels = len(label_filter)
    fig, axs = plt.subplots(num_labels, 2, sharex='col', figsize=(6, num_labels * 2),
                            dpi=150, )  # gridspec_kw={'top':.9})
    fig.suptitle('Performance Metrics for\n{}'.format(filename))

    # Initialize values for each corestat
    thresholds = []
    mid_thresh = []
    max_score = []
    for corestat in corestats:
        thresholds.append(sorted(corestat.keys()))
        mid_thresh.append(thresholds[-1][int(len(thresholds[-1]) / 2)])
        max_score.append(np.max([_get_classes_max(num_classes, corestat, mid_thresh[-1], 'tp'),
                                 _get_classes_max(num_classes, corestat, mid_thresh[-1], 'fp'),
                                 _get_classes_max(num_classes, corestat, mid_thresh[-1], 'n_gt')]))
    axs[0][0].set_title('Recall-Accuracy')
    axs[0][1].set_title('Samples for thresh = %.2f' % mid_thresh[0])
    curr_ax = 0
    for cls in range(1, num_classes + 1):
        if cls in label_filter:
            # Subplots for recall-accuracy curves
            ax = axs[curr_ax]
            sorted_ap = [val[cls] for val in ap]
            keys = np.argsort(sorted_ap[::-1])
            for idx, key in enumerate(keys):
                _plot_acc_rec(ax[0], corestats[key], cls, thresholds[key], COLORS[key],
                              mid_thresh[0])
                ax[0].text(0.95, 0.86 - idx * 0.1, '$AP: %.2f$' % (ap[key][cls]), size=9,
                           color=COLORS[key],
                           horizontalalignment='right',
                           verticalalignment='center', transform=ax[0].transAxes)
            ax[0].set_ylabel(labels[cls]['name'])
            ax[0].set_xlim(0.0, 1.0)
            ax[0].set_ylim(0.0, 1.0)

            # Subplots for tp, fp and gt scores
            for idx, key in enumerate(keys):
                _plot_bar_chart(ax[1], corestats[key], cls, mid_thresh[key], max_score[key],
                                COLORS[key], relative_bar_chart)
            bar_labels = ['Tp', 'Fp', 'GT']
            positions = np.arange(3) + 1
            ax[1].set_yticks(positions)
            ax[1].set_yticklabels(bar_labels)
            ax[1].set_ylim(0, 4)
            ax[1].spines['top'].set_color(None)
            ax[1].spines['right'].set_color(None)
            ax[1].text(0.98, 0.92, '$n_{%s} = %d$' % (
                labels[cls]['name'], corestats[0][mid_thresh[0]]['n_gt'][cls]), size=9,
                       horizontalalignment='right',
                       verticalalignment='center', transform=ax[1].transAxes)
            curr_ax += 1
    fig.tight_layout(rect=[0, 0, 1, 0.81 + 0.02 * num_labels])
    return fig


def _plot_acc_rec(ax, corestats, cls, thresholds, color, mid_thresh):
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
    for thresh in thresholds[::-1]:
        acc.append(corestats[thresh]['acc'][cls])
        rec.append(corestats[thresh]['rec'][cls])
        if corestats[thresh]['acc_ci'][cls] is not None:
            _acc_ci.append(corestats[thresh]['acc_ci'][cls])
            _rec_ci.append(corestats[thresh]['rec_ci'][cls])
            acc_low.append(acc[-1] - _acc_ci[-1])
            acc_high.append(acc[-1] + _acc_ci[-1])
            rec_low.append(rec[-1] - _rec_ci[-1])
            rec_high.append(rec[-1] + _rec_ci[-1])
    rec_mid, acc_mid = corestats[mid_thresh]['rec'][cls], corestats[mid_thresh]['acc'][cls]
    ax.stackplot(rec, acc, color=color, alpha=ALPHA, zorder=2)
    ax.plot(rec, acc, '--', color=color, lw=1.5, zorder=3)

    # if the conf bounds have been filled
    if acc_low:
        ax.plot(rec_low, acc_low, ':', color=color, lw=1.5, zorder=3)
        ax.plot(rec_high, acc_high, ':', color=color, lw=1.5, zorder=3)
    ax.plot(rec_mid, acc_mid, color='dark' + color, marker='x', ms=6, mew=1.5, fillstyle='none',
            zorder=10)


def _plot_bar_chart(ax, corestats, cls, mid_thresh, max_score, color, relative_bar_chart):
    tp = corestats[mid_thresh]['tp'][cls]
    fp = corestats[mid_thresh]['fp'][cls]
    n_gt = corestats[mid_thresh]['n_gt'][cls]
    if relative_bar_chart:
        tp = tp / n_gt
        fp = fp / n_gt
        n_gt = 1.0
        ax.set_xlim(0.0, 1.0)
    else:
        ax.set_xlim(0, max_score)
    scores = [tp, fp, n_gt]
    positions = np.arange(3) + 1
    ax.barh(positions, scores, 0.5, align='center', color=color, edgecolor='black', lw=1,
            alpha=ALPHA)


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
