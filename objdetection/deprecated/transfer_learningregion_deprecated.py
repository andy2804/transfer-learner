import os

import cv2
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib import ticker
from matplotlib.backends.backend_pdf import PdfPages

from objdetection.deprecated import input_formatter_deprecated, load_recording_deprecated

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

if __name__ == '__main__':
    # ================= Inintialization ===========================#
    # folders with recordings
    recs_path = "/media/ale/ZILLY_DISK/ALESSANDRO/zuriscapes_official/TRAIN"
    recordings = [os.path.join(recs_path, rec_name) for rec_name
                  in os.listdir(recs_path) if
                  rec_name == "lcmlog-2017-11-13.03"]
    # select one
    rec = recordings[0]
    loader = load_recording_deprecated.Loader
    events_dict = loader.load_events(fold=rec)
    frames_list = loader.load_frames(fold=rec)
    # add absolute path
    frames_list = [(os.path.join(rec, f[0]), f[1]) for f in frames_list]
    formatter_sae_gaus = input_formatter_deprecated.SAE("_11gaus")
    # ================= Extract what to plot ===================== #
    frame_num = 153
    end_ts = frames_list[frame_num][1]  # [s]
    span_ts = 0.005  # [s]
    begin_ts = end_ts - span_ts
    enanched = True
    assert begin_ts >= 0
    #
    image_path = frames_list[frame_num][0]
    image_np = cv2.imread(image_path, 0)
    if enanched:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image_np = clahe.apply(image_np)
    #
    events = formatter_sae_gaus.crop_and_format_events(
            events_dict, frame_ts=end_ts, previous_ts=begin_ts)
    # ================= Plotting ===================== #
    fig = plt.figure("image")
    fig.subplots_adjust(wspace=0, hspace=0)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.xaxis.set_major_locator(ticker.NullLocator())
    ax1.yaxis.set_major_locator(ticker.NullLocator())
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.xaxis.set_major_locator(ticker.NullLocator())
    ax2.yaxis.set_major_locator(ticker.NullLocator())
    ax1.imshow(image_np, cmap='gray')
    ax2.imshow(events, cmap='gray')
    plt.show()
    # ================= Storing ===================== #
    with PdfPages('/home/ale/Pictures/foo_temp.pdf') as pdf:
        # As many times as you like, create a figure fig and save it:
        pdf.savefig(figure=fig, bbox_inches='tight')
