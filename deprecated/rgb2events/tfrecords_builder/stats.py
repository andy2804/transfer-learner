"""
author: aa & az
"""
import os
import pickle
from collections import namedtuple

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from objdetection.meta.utils_generic.magic_constants import DAVIS240c
from objdetection.meta.utils_labeler.static_helper import load_labels

ObjStats = namedtuple('ObjStats', ['label',
                                   'ymin',
                                   'xmin',
                                   'h',
                                   'w',
                                   'tl_score',
                                   'tl_difficult'])


class TLStatistician:
    def __init__(self,
                 tl_score_threshold,
                 norm_wrt=(DAVIS240c.height, DAVIS240c.width),
                 labels_file='zauron_label_map.json'):
        self._labels = load_labels(labels_file)
        self._n_instances = 0
        self._objstats = []
        self._norm_wrt = norm_wrt
        self._tl_score_thresh = tl_score_threshold
        self._keep_score_label_map = {'None': 'All', 'False': 'Difficult', 'True': 'Approved'}

    def append_obj_stats(self, label, ymin, xmin, h, w, tl_score, tl_difficult):
        """
        :param label:
        :param ymin: normalized coord
        :param xmin: normalized coord
        :param h: normalized coord
        :param w: normalized coord
        :param tl_score:
        :param tl_difficult:
        :return:
        """
        self._objstats.append(ObjStats(
                label=label, ymin=ymin, xmin=xmin, h=h, w=w, tl_score=tl_score,
                tl_difficult=tl_difficult))

    def get_obj(self, label_filt=None, tl_keep_filt=None):
        return [obj for obj in self._objstats if
                (tl_keep_filt is None or tl_keep_filt == obj.tl_difficult) and
                (label_filt is None or label_filt == obj.label)]

    def get_avg_area(self, label_filt=None, tl_keep_filt=None):
        """
        :param label_filt:
        :param tl_keep_filt: set to 'True' or 'False' if the stats should be computed only for
        objects with a specific transfer learning region flag
        :return:
        """
        areas = np.array(
                [obj.h * self._norm_wrt[0] * obj.w * self._norm_wrt[1] for obj in self._objstats if
                 (tl_keep_filt is None or tl_keep_filt == obj.tl_difficult) and
                 (label_filt is None or label_filt == obj.label)])
        return np.mean(areas), np.std(areas), np.max(areas), np.min(areas)

    def get_tlscores(self, label_filt=None, tl_keep_filt=None):
        """
        :param label_filt:
        :param tl_keep_filt: set to 'True' or 'False' if the stats should be computed
        only for
        objects with a specific transfer learning region flag
        :return:
        """
        scores = np.array([obj.tl_score for obj in self._objstats if
                           (tl_keep_filt is None or tl_keep_filt == obj.tl_difficult) and
                           (label_filt is None or label_filt == obj.label)])
        return scores

    def get_aspect_ratios(self, label_filt=None, tl_keep_filt=None):
        """
        :param label_filt:
        :param tl_keep_filt: set to 'True' or 'False' if the stats should be computed
        only for
        objects with a specific transfer learning region flag
        :return:
        """
        asp_ratios = np.array(
                [(obj.w * self._norm_wrt[1]) / (obj.h * self._norm_wrt[0])
                 for obj in self._objstats if
                 (tl_keep_filt is None or tl_keep_filt == obj.tl_difficult) and
                 (label_filt is None or label_filt == obj.label)])

        return asp_ratios

    def num_obj(self, label_filt=None, tl_keep_filt=None):
        """
        :param label_filt:
        :param tl_keep_filt: set to 'True' or 'False' if the stats should be computed
        only for
        objects with a specific transfer learning region flag
        :return:
        """
        count = 0
        for obj in self._objstats:
            if (tl_keep_filt is None or tl_keep_filt == obj.tl_difficult) and (
                    label_filt is None or label_filt == obj.label):
                count += 1
        return count

    def save(self, output_dir, filename):
        file = os.path.join(output_dir, filename + '_objstats.pickle')
        with open(file, 'wb') as fp:
            pickle.dump(self._objstats, fp)
        pass

    def make_plots(self, save_plots=False, output_dir='', filename='', show_plots=False,
                   labels_dict=None):
        # keep it modular, _make_plot_1(),_make_plot_2(), _make_plot_3()...
        output_file = os.path.join(output_dir, filename)
        if labels_dict is None:
            labels_dict = self._labels
        self._set_fonts()
        # self._heatmap_of_detections()
        # self._full_heatmap_of_detections()

        self._full_dist_of_aspect_ratio(labels=labels_dict, save_plots=save_plots,
                                        output_file=output_file, show_plots=show_plots)
        self._full_dist_of_tlscore(labels=labels_dict, save_plots=save_plots,
                                   output_file=output_file, show_plots=show_plots)
        return

    def _full_dist_of_aspect_ratio(self, labels, save_plots, output_file, show_plots):
        n_categories = len(labels)
        fig, axn = plt.subplots(n_categories, 3,
                                sharex='col',
                                sharey='row',
                                figsize=(6, 8),
                                dpi=100)
        fig.suptitle("Aspect Ratios dist.", size=18)
        for row_idx, row_ax in enumerate(axn):
            label_filt = row_idx + 1
            cat = labels[label_filt]["name"]
            for col_idx, col_ax in enumerate(row_ax):
                if col_idx % 3 == 1:
                    tl_keep_filt = False
                elif col_idx % 3 == 2:
                    tl_keep_filt = True
                else:
                    tl_keep_filt = None
                data = self.get_aspect_ratios(label_filt=label_filt, tl_keep_filt=tl_keep_filt)
                if data.size > 0:
                    sns.distplot(data, kde=False, ax=col_ax,
                                 hist_kws={"histtype":  "stepfilled",
                                           "linewidth": 3, "alpha": 1,
                                           "color":     "g"
                                           })
                # don't move these before sns
                if col_idx % 3 == 0:
                    col_ax.set_ylabel("{}".format(cat), rotation=90)
                if row_idx == 0:
                    col_ax.set_title("{}".format('All' if tl_keep_filt is None else tl_keep_filt))

        fig.tight_layout(rect=[0, 0, 1, .95])
        if save_plots:
            fig.savefig(output_file + '_stats_aspectratios.pdf')
        if show_plots:
            plt.show()
        return

    def _full_dist_of_tlscore(self, labels, save_plots, output_file, show_plots):
        n_categories = len(labels)
        fig, axn = plt.subplots(n_categories, 3, sharex='col', sharey='row', figsize=(8, 10),
                                dpi=150)
        fig.suptitle("Transfer Learning Score dist.", size=18)
        for row_idx, row_ax in enumerate(axn):
            label_filt = row_idx + 1
            cat = labels[label_filt]["name"]
            for col_idx, col_ax in enumerate(row_ax):
                if col_idx % 3 == 1:
                    tl_keep_filt = False
                elif col_idx % 3 == 2:
                    tl_keep_filt = True
                else:
                    tl_keep_filt = None
                data = self.get_tlscores(label_filt=label_filt, tl_keep_filt=tl_keep_filt)
                if data.size > 0:
                    hist_kws = {"histtype": "stepfilled", "linewidth": 3, "alpha": 1}
                    if tl_keep_filt is False:
                        data_lowscore = data[data <= self._tl_score_thresh]
                        data_toosmall = data[data > self._tl_score_thresh]
                        sns.distplot(data_lowscore,
                                     kde=False, ax=col_ax, color='g', hist_kws=hist_kws)
                        sns.distplot(data_toosmall,
                                     kde=False, ax=col_ax, color='r', hist_kws=hist_kws)
                        col_ax.legend(
                                ["{}".format(len(data_lowscore)), "{}".format(len(data_toosmall))])
                    else:
                        sns.distplot(data,
                                     kde=False, ax=col_ax, color='g', hist_kws=hist_kws)
                        col_ax.legend(["{}".format(len(data))])
                # don't move these before sns
                if col_idx % 3 == 0:
                    col_ax.set_ylabel("{}".format(cat))
                if row_idx == 0:
                    col_ax.set_title("{}".format(self._keep_score_label_map[str(tl_keep_filt)]))

        fig.tight_layout(rect=[0, 0, 1, .95])
        plt.legend()
        if save_plots:
            fig.savefig(output_file + '_stats_tlscore.pdf')
        if show_plots:
            plt.show()
        return

    def _heatmap_of_detections(self, label_filt=None, tl_keep_filt=None):
        data = np.zeros(self._norm_wrt)
        objs = self.get_obj(label_filt=label_filt, tl_keep_filt=tl_keep_filt)
        for obj in objs:
            ymin, xmin = int(obj.ymin * self._norm_wrt[0]), int(obj.xmin * self._norm_wrt[1])
            h, w = int(obj.h * self._norm_wrt[0]), int(obj.xmin * self._norm_wrt[1])
            data[ymin:ymin + h, xmin:xmin + w] += 1
        ax = sns.heatmap(data, xticklabels=False, yticklabels=False,
                         cbar_kws={"orientation": "horizontal"})
        cat = 'global' if label_filt is None else self._labels[label_filt]["name"]
        ax.set_title('Heatmap of detections of label: %s' % cat)
        return

    def _full_heatmap_of_detections(self, ):
        fig = plt.figure(figsize=(6, 8), dpi=100)
        fig.suptitle("Heatmap of detections", size=18)
        n_categories = len(self._labels)
        gs = GridSpec(n_categories, 3)
        gs.update(bottom=0.1, top=0.9)
        gx_cbar = GridSpec(1, 1)
        gx_cbar.update(bottom=0.03, top=0.07)
        axn = [[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(n_categories)]
        cbar_ax = fig.add_subplot(gx_cbar[0, 0])
        for row_idx, row_ax in enumerate(axn):
            label_filt = row_idx + 1
            cat = self._labels[label_filt]["name"]
            for col_idx, col_ax in enumerate(row_ax):
                if col_idx % 3 == 1:
                    tl_keep_filt = False
                elif col_idx % 3 == 2:
                    tl_keep_filt = True
                else:
                    tl_keep_filt = None
                objs = self.get_obj(label_filt=label_filt, tl_keep_filt=tl_keep_filt)
                data = np.zeros(self._norm_wrt)
                for obj in objs:
                    ymin, xmin = int(obj.ymin * self._norm_wrt[0]), int(
                            obj.xmin * self._norm_wrt[1])
                    h, w = int(obj.h * self._norm_wrt[0]), int(obj.xmin * self._norm_wrt[1])
                    data[ymin:ymin + h, xmin:xmin + w] += 1
                sns.heatmap(data, ax=col_ax, xticklabels=False, yticklabels=False,
                            cbar=True, cbar_ax=cbar_ax,
                            cbar_kws={"orientation": "horizontal"})
                # todo set vmin vmax for common scale of plots, also cbar needed only once?!
                # don't move these before sns
                if col_idx % 3 == 0:
                    col_ax.set_ylabel("{}".format(cat))
                if row_idx == 0:
                    col_ax.set_title("{}".format('All' if tl_keep_filt is None else tl_keep_filt))

        # plt.tight_layout()
        plt.show()
        # plt.savefig("")
        return

    @staticmethod
    def _set_fonts():
        plt.rc('text', usetex=True)
        font = {'family': 'serif',
                'weight': 'bold',
                'size':   12
                }
        plt.rc('font', **font)

    @property
    def n_instances(self):
        return self._n_instances

    @n_instances.setter
    def n_instances(self, value):
        self._n_instances = value

    def reset_n_instances(self):
        self._n_instances = 0

    def reset_objstats(self):
        self._objstats = []
