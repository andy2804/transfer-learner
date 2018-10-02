import matplotlib.patches as mpatches
import numpy as np
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

if __name__ == '__main__':
    # Example data
    classes = ('truck', 'bicycle', 'person', 'bus/tram', 'car', 'motorcycle')
    y_pos = np.arange(len(classes))
    bar_width = 0.35
    performance_our = np.array([0.13, 0.27, 0.10, 0.54, 0.19, 0.11]) * 100
    performance_google = np.array([0.3, 0.08, 0.11, 0.46, 0.19, 0.01]) * 100
    error = 0.01 * np.random.rand(len(classes))
    #

    # Actual visualization
    fig, ax = plt.subplots()
    ax.barh(y_pos, performance_our, bar_width, align='center',
            color='blue', xerr=error)
    ax.barh(y_pos + bar_width, performance_google, bar_width, align='center',
            color='orange', xerr=error)
    green_patch = mpatches.Patch(color='blue', label='event-based')
    yellow_patch = mpatches.Patch(color='orange', label='frame-based')
    plt.legend(handles=[green_patch, yellow_patch], fontsize=14)
    ax.set_yticks(y_pos + bar_width / 2)
    ax.set_yticklabels(classes, **axis_font)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('percent. \%', **axis_font)
    ax.set_title('Recall', **title_font)
    ax.set_xlim([0, 60])

    plt.show()

    with PdfPages('/home/ale/Pictures/foo_perf_temp.pdf') as pdf:
        # As many times as you like, create a figure fig and save it:
        pdf.savefig(figure=fig, bbox_inches='tight')
