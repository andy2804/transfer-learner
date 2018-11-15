import os
import random
from collections import namedtuple

import cv2
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib import ticker
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import proj3d

from objdetection.deprecated import input_formatter_deprecated, load_recording_deprecated

# import sys
# sys.path.append(os.getcwd()[:os.getcwd().index('NeuromorphicDeepLearning')])

Options = namedtuple('Options', ["camera", "X", "Y"])
opt_default = Options("davis240c", 240, 180)
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)


def make_get_proj(self, rx, ry, rz):
    """
    Return a variation on :func:`~mpl_toolkit.mplot2d.axes3d.Axes3D.getproj` 
    that
    makes the box aspect ratio equal to *rx:ry:rz*, using an axes object *self*.
    """
    rm = max(rx, ry, rz)
    kx = rm / rx
    ky = rm / ry
    kz = rm / rz

    # Copied directly from mpl_toolkit/mplot3d/axes3d.py. New or modified 
    # lines are
    # marked by ##
    def get_proj():
        relev, razim = np.pi * self.elev / 180, np.pi * self.azim / 180

        xmin, xmax = self.get_xlim3d()
        ymin, ymax = self.get_ylim3d()
        zmin, zmax = self.get_zlim3d()

        # transform to uniform world coordinates 0-1.0,0-1.0,0-1.0
        worldM = proj3d.world_transformation(xmin, xmax,
                                             ymin, ymax,
                                             zmin, zmax)

        # adjust the aspect ratio                          ##
        aspectM = proj3d.world_transformation(-kx + 1, kx,  ##
                                              -ky + 1, ky,  ##
                                              -kz + 1, kz)  ##

        # look into the middle of the new coordinates
        R = np.array([0.5, 0.5, 0.5])

        xp = R[0] + np.cos(razim) * np.cos(relev) * self.dist
        yp = R[1] + np.sin(razim) * np.cos(relev) * self.dist
        zp = R[2] + np.sin(relev) * self.dist
        E = np.array((xp, yp, zp))

        self.eye = E
        self.vvec = R - E
        self.vvec = self.vvec / proj3d.mod(self.vvec)
        if abs(relev) > np.pi / 2:
            # upside down
            V = np.array((0, 0, -1))
        else:
            V = np.array((0, 0, 1))
        zfront, zback = -self.dist, self.dist
        viewM = proj3d.view_transformation(E, R, V)
        perspM = proj3d.persp_transformation(zfront, zback)
        M0 = np.dot(viewM, np.dot(aspectM, worldM))
        M = np.dot(perspM, M0)
        return M

    return get_proj


class PlotterEventBased:
    def __init__(self, option=None):
        if isinstance(option, Options):
            self.opt = option
        else:
            self.opt = opt_default
        self.fig = plt.figure()  # plt.figure(figsize=(18, 24))

    def plot2Devents(self, events2plot, ax=None):
        minusoneone_to_0255 = lambda x: np.multiply(
                np.stack([1 + x] * 3, axis=-1), 255 / 2).astype(dtype=np.uint8)
        im2plot = list(map(minusoneone_to_0255, [events2plot]))
        if ax is None:
            ax = self.fig.gca()
        ax.imshow(im2plot[0], cmap='gray')
        gs0 = gridspec.GridSpec(1, 2)
        gs0.update(wspace=0.0, hspace=0.0)
        ax.xaxis.set_major_locator(ticker.NullLocator())
        ax.yaxis.set_major_locator(ticker.NullLocator())
        ax.set_xbound(0, self.opt.X - 1)
        ax.set_ybound(0, self.opt.Y - 1)
        ax.set_frame_on(True)
        return

    def plot3Devents(self, events, frames=None, ax=None):
        """
        good example: Recordings_static/20170925T184857_de60f5fb.lcm.00
        :param frames: (frame, ts_bin?!)
        :param events: (time, Y, X, channels)
        :return:
        """
        if ax is None:
            ax = self.fig.add_subplot(111, projection='3d')
        if frames is not None:
            Y = np.arange(0, self.opt.X, 1)
            Z = np.arange(0, self.opt.Y, 1)
            Y, Z = np.meshgrid(Y, Z)
            for frame in frames:
                image_np = cv2.imread(frame[0], 0)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                image_np = clahe.apply(image_np)
                X = frame[1]
                img = image_np / 255
                col = np.dstack([img] * 3).copy(order='C')
                ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=col)
        # todo speed up plot_surface and multiple frames fix
        if events.shape[3] == 1:
            time, y, x = events.nonzero()[0:3]
            ax.scatter(time, x, y, c='red', marker='.')
        else:
            time, y, x = events[:, :, :, 0].nonzero()[0:3]
            ax.scatter(time, x, y, c='red', marker='.')
            time, y, x = events[:, :, :, 1].nonzero()[0:3]
            ax.scatter(time, x, y, c='blue', marker='.')
        ax.set_xlabel('time [ms]')
        # ax.set_ylabel('x')
        # ax.set_zlabel('y')
        ax.set_xbound(0, events.shape[0])
        ax.set_ybound(0, self.opt.X - 1)
        ax.set_zbound(0, self.opt.Y - 1)
        ax.xaxis.set_major_locator(ticker.NullLocator())
        ax.yaxis.set_major_locator(ticker.NullLocator())
        ax.zaxis.set_major_locator(ticker.NullLocator())
        ax.set_frame_on(True)
        ax.invert_zaxis()
        ax.get_proj = make_get_proj(ax, 3, 1.8, 2.4)
        ax.set_aspect(1.0)
        return

    def add_equal_subplots(self, rows, columns, axis_proj):
        return [self.fig.add_subplot(rows, columns, num_subplot + 1,
                                     projection=axis_proj[num_subplot]) for
                num_subplot in range(rows * columns)]

    def add_manually_subplots(self, ):
        ax_list = [self.fig.add_subplot(1, 2, 1),
                   self.fig.add_subplot(1, 2, 2),
                   # self.fig.add_subplot(1, 1, 1, projection='3d')
                   # self.fig.add_subplot(2, 2, 3),
                   # self.fig.add_subplot(2, 2, 4),
                   ]
        return ax_list

    @staticmethod
    def load_single_image(image_path, enanched=False):
        image_np = cv2.imread(image_path, 0)
        if enanched:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image_np = clahe.apply(image_np)
        image_np = image_np / (255 / 2) - 1
        return image_np

    def display(self, ):
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=0, hspace=0)
        plt.show()
        return


if __name__ == "__main__":
    recs_path = "/home/ale/encoder/zuriscapes/export"
    # select one at random?
    recordings = [os.path.join(recs_path, rec_name) for rec_name in
                  os.listdir(recs_path)]
    rec = random.choice(recordings)
    # not at random ->
    rec = recordings[15]
    # load and plot
    loader = load_recording_deprecated.Loader
    plotter = PlotterEventBased()
    # formatter_sae_exp = input_formatter.SAE("_11exp")
    formatter_sae_gaus = input_formatter_deprecated.SAE("_11gaus")
    # formatter_3d = input_formatter.THREED()
    # formatter_sum_sat = input_formatter.SUM('saturated_sum')
    # formatter_sum_plain = input_formatter.SUM('plain_sum')
    events_dict = loader.load_events(fold=rec)
    frames_list = loader.load_frames(fold=rec)
    # add absolute path
    frames_list = [[os.path.join(rec, f[0]), f[1]] for f in frames_list]
    # ======= Compute ts from which we crop events and other parameters
    frame_number = 3
    time_bins = 500
    end_ts = frames_list[frame_number][1]  # [s]
    span_ts = 0.03  # [s]
    begin_ts = end_ts - span_ts
    assert begin_ts >= 0

    # events_01 = formatter_sum_sat.crop_and_format_events(
    # 		events_dict, frame_ts=end_ts, previous_ts=begin_ts)
    # events_02 = formatter_sum_sat.crop_and_format_events(
    # 		events_dict, frame_ts=end_ts, previous_ts=begin_ts-span_ts*5)
    events_03 = formatter_sae_gaus.crop_and_format_events(
            events_dict, frame_ts=end_ts, previous_ts=begin_ts)

    # events_04 = formatter_sum_plain.crop_and_format_events(
    # 		events_dict, frame_ts=end_ts, previous_ts=begin_ts)

    # events_03 = formatter_3d.crop_and_format_events(events_dict, 
    # frame_ts=end_ts,
    #                                                previous_ts=begin_ts, 
    # time_bins=time_bins)
    # frames_03 = formatter_3d.crop_and_format_frames(frames_list, 
    # frame_ts=end_ts,
    #                                                previous_ts=begin_ts, 
    # time_bins=time_bins)
    axis_list = plotter.add_manually_subplots()
    image = plotter.load_single_image(frames_list[frame_number][0],
                                      enanched=True)
    plotter.plot2Devents(image, axis_list[0])
    axis_list[0].set_title("frame")
    axis_list[1].set_title("events")
    plotter.plot2Devents(events_03, axis_list[1])
    # plotter.plot2Devents(events_04, axis_list[3])
    # plotter.plot3Devents(events_03, frames=frames_03, ax=axis_list[0])
    with PdfPages('/home/ale/Pictures/foo_temp.pdf') as pdf:
        # As many times as you like, create a figure fig and save it:
        pdf.savefig(plotter.fig)
    # plotter.fig.savefig('/home/ale/Pictures/foo_temp.png', bbox_inches='tight')
    plotter.display()
