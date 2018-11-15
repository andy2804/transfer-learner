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

if __name__ == '__main__':
    # Example data
    rec_ev = [0.0,
              0.0033548387096774194,
              0.012903225806451613,
              0.035612903225806451,
              0.052387096774193551,
              0.090064516129032254,
              0.14477419354838711,
              0.21083870967741936,
              0.2627096774,
              0.3]
    acc_ev = [1.0,
              1.0,
              1.0,
              0.9928057553956835,
              0.97596153846153844,
              0.86815920398009949,
              0.61716171617161719,
              0.26474400518470514,
              0.05199713965,
              0.02]
    rec_im = [0.0,
              0.06322580645161291,
              0.08645161290322581,
              0.10193548387096774,
              0.11148387096774194,
              0.12232258064516129,
              0.1336774193548387,
              0.1550967741935484,
              0.17909677419354839,
              0.20412903225806453,
              0.2449032258,
              0.29316129032258,
              0.3401290322580645,
              0.52,
              0.6]
    acc_im = [0.99,
              0.9459459459459459,
              0.9103260869565217,
              0.8797327394209354,
              0.8355899419729207,
              0.7886855241264559,
              0.7368421052631579,
              0.6715083798882682,
              0.6082383873794917,
              0.5210803689064558,
              0.3957464554,
              0.2402707275803722,
              0.14568365203935,
              0.05903377963,
              0.03]
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
    ax.set_xlabel('recall', **axis_font)
    ax.set_ylabel('accuracy', **axis_font)
    ax.set_title('mAP', size=20)
    ax.set_xlim([0, .6])
    ax.set_ylim([0, 1])

    plt.show()

    with PdfPages('/home/ale/Pictures/foo_perf_temp.pdf') as pdf:
        # As many times as you like, create a figure fig and save it:
        pdf.savefig(figure=fig, bbox_inches='tight')
