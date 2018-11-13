"""
author: az
"""
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.backends.backend_pdf import PdfPages

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
title_font = {'fontname':          'Arial', 'size': '25', 'color': 'black',
              'weight':            'heavy',
              'verticalalignment': 'bottom'
              }
axis_font = {'fontname': 'Arial', 'size': '18'}

labels = "event-based"
# Actual visualization
fig, ax = plt.subplots()

ax.stackplot(rec_im, acc_im, color='orange', alpha=1)
ax.stackplot(rec_ev, acc_ev, color='blue', alpha=1)
green_patch = mpatches.Patch(color='blue', label='event-based')
red_patch = mpatches.Patch(color='orange', label='frame-based')
plt.legend(handles=[green_patch, red_patch], fontsize=16)
# ax.set_yticks(y_pos + bar_width/2)
# ax.set_yticklabels(classes)
ax.invert_yaxis()  # labels read top-to-bottom

plt.show()


def plot_tp_fp_fn(corestats, labels_map):
    tp = [corestats['tp'][i] for i in corestats['tp']]
    fp = [corestats['fp'][i] for i in corestats['fp']]
    n_gt = [corestats['n_gt'][i] for i in corestats['n_gt']]

    fig = plt.figure()
    plt.legend()


def plot_overall_map(corestats, equal_class_weight=False, pdf_file=None):
    ax.set_xlabel('recall', **axis_font)
    ax.set_ylabel('accuracy', **axis_font)
    ax.set_title('mAP', **title_font)
    ax.set_xlim([0, .6])
    ax.set_ylim([0, 1])
    if pdf_file is not None:
        _save_to_pdf(fig, pdf_file)
    return


def _save_to_pdf(fig, pdf_file):
    try:
        with PdfPages('/home/ale/Pictures/foo_perf_temp.pdf') as pdf:
            pdf.savefig(figure=fig, bbox_inches='tight')
    except IOError:
        print("Failed to pdf failed!")
    return
