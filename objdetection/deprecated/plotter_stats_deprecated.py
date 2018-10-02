import numpy as np
from matplotlib import pyplot as plt


# todo deprecated
def plot_stats(stats_log):
    print("Plotting...")
    # Unfolding stats
    aspect_ratios = np.asarray([sample["aspect_ratio"] for sample in stats_log])
    aspect_ratios = np.concatenate(aspect_ratios).ravel()
    volumes = np.asarray([sample["volume"] for sample in stats_log])
    volumes = np.concatenate(volumes).ravel()
    # Plotting
    bins = np.concatenate([np.arange(0, 1, 0.1), np.arange(1, 11, 1)])
    fig = plt.figure()
    # subplot 1
    ax1 = fig.add_subplot(211)
    ax1.hist(aspect_ratios, bins=bins)
    # TODO FIX SCALING OF THE AXIS
    ax1.set_xticks(bins)
    ax1.set_title("Aspect ratios")
    # subplot 2
    ax2 = fig.add_subplot(212)
    ax2.hist(volumes, bins='auto')
    ax2.set_title("Volumes of objects")
    plt.show()
