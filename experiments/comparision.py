"""This python module implements the class ``ClassifierComparision``
which can be used to compare the accuracy results of two different
classification results (e.g. results from different fruits.Fruit
objects).

This file can also be used as a scripted invoked from the command line.
You get all available arguments with

    >>> python comparision.py -h

The module can also be used without any dependencies to fruits.
"""

import os
import argparse
from typing import Union

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

DEFAULT_COMPARISION_COLUMN = "FRUITS Acc"

_COLORS: list[tuple[int, int, int]] = [
    (0, 100, 173),
    (181, 22, 33),
    (87, 40, 98),
    (0, 48, 100),
    (140, 25, 44),
]


def _get_color(i: int) -> tuple[float, float, float]:
    if i < 5:
        return tuple(x/255 for x in _COLORS[i])
    elif i < 20:
        return cm.get_cmap("tab20b").colors[2:][i-5]
    else:
        return tuple(x/255 for x in _COLORS[i % len(_COLORS)])


class ClassifierComparision:
    """Implements methods for the comparision of two classification
    techniques using the information of their accuracy on different
    datasets.

    :param acc1: A one dimensional numpy array containing accuracy
        results of one technique (the one that is expected to be
        better in general) for different datasets.
    :type acc1: np.ndarray
    :param acc2:  A one dimensional numpy array containing accuracy
        results of a second technique.
    :type acc2: np.ndarray
    :param label1: Short label that describes the first technique.
    :type label1: str
    :param label2:  Short label that describes the second technique.
    :type label2: str
    """

    def __init__(
        self,
        accuracies: np.ndarray,
        labels: list[str],
    ):
        self._ndatasets = accuracies.shape[0]
        self._nclassifiers = accuracies.shape[1]
        if len(labels) != self._nclassifiers:
            raise ValueError("Lengths of accuracies and labels differ")
        self._accuracies = accuracies.copy()
        maximum = self._accuracies.max()
        if maximum > 1.0:
            self._accuracies /= maximum
        self._labels = labels

    def scatterplot(
        self,
        indices: Union[list[tuple[int, int]], None] = None,
        opacity: Union[list[float], None] = None,
    ) -> tuple:
        """Creates a 2D scatter plot for each pair of the given
        accuracy results.

        :param indices: List of integer pairs that define which methods
            to compare. If ``None`` is given, then all plots will be
            compared.
        :type indices: Union[List[Tuple[int]], None], optional
        :param opacity: List of floats that has the same length as
            the original accuracy results. The points in the scatter
            plot will be colored based on the values in this list.,
            defaults to None
        :type opacity: Union[List[float], None], optional
        :returns: Figure and axis that you get from ``plt.subplots``.
        :rtype: tuple
        """
        colors = np.zeros((self._ndatasets, 4))
        colors[:, :3] = _get_color(0)
        colors[:, 3] = opacity
        if indices is None:
            indices = [(i, j)
                       for i in range(self._nclassifiers)
                       for j in range(self._nclassifiers)]
            fig, axs = plt.subplots(self._nclassifiers, self._nclassifiers)
        else:
            fig, axs = plt.subplots(len(indices), 1)
            if len(indices) == 1:
                axs = np.array([axs], dtype=object)
            axs = axs.reshape((len(indices), 1))
        c = 0
        for i in range(axs.shape[0]):
            for j in range(axs.shape[1]):
                ii, jj = indices[c]
                axs[i][j].axis('square')
                axs[i][j].set_xlim([0, 1])
                axs[i][j].set_ylim([0, 1])
                if ii == jj:
                    weights = np.ones_like(self._accuracies[:, ii])
                    weights /= self._ndatasets
                    axs[i][j].hist(
                        self._accuracies[:, ii],
                        weights=weights,
                    )
                else:
                    axs[i][j].scatter(
                        self._accuracies[:, jj], self._accuracies[:, ii],
                        c=opacity,
                        cmap="copper_r",
                    )

                    axs[i][j].plot([0, 1], [0, 1],
                                   transform=axs[i][j].transAxes,
                                   color=_get_color(1), ls="--")
                    axs[i][j].plot([0.05, 1], [0, 0.95],
                                   transform=axs[i][j].transAxes,
                                   color=_get_color(1) + (0.3,), ls="--")
                    axs[i][j].plot([0, 0.95], [0.05, 1],
                                   transform=axs[i][j].transAxes,
                                   color=_get_color(1) + (0.3,), ls="--")

                    meanii = self._accuracies[:, ii].mean()
                    meanjj = self._accuracies[:, jj].mean()
                    axs[i][j].axhline(meanii, xmin=0, xmax=meanii,
                                      color=_get_color(3) + (0.5,), ls="--")
                    axs[i][j].axvline(meanjj, ymin=0, ymax=meanjj,
                                      color=_get_color(3) + (0.5,), ls="--")

                    axs[i][j].text(0.02, 0.98, self._labels[ii],
                                   size="large", ha="left", va="top")
                    axs[i][j].text(0.98, 0.02, self._labels[jj],
                                   size="large", ha="right", va="bottom")
                c += 1
        return fig, axs

    def test_greater(self, i: int, j: int):
        """Tests whether the null-hypothesis of technique at index ``i``
        being less or equally good compared to method ``j`` can be
        rejected by performing an one-sided paired Wilcoxon signed-rank
        test.

        :type i: int
        :type j: int
        :returns: Value of the test function and p-value of the test.
        :rtype: tuple
        """
        stat, p = stats.wilcoxon(
            self._accuracies[:, i],
            self._accuracies[:, j],
            alternative="greater",
        )
        return stat, p

    def critical_difference_diagram(self, alpha: float = 0.05):
        """Draws and returns a figure of a critical difference diagram
        based on the accuracies given to the class object.
        This type of plot was described in the paper
        'Statistical Comparision of Classifiers over Multiple Data Sets'
        by Janez Demsar, 2006.

        :param alpha: Significance value used for doing pairwise
            Wilcoxon signed-rank tests., defaults to 0.05
        :type alpha: float, optional
        :returns: Figure and axis that matches to the return types of
            ``plt.subplots(1, 1)``.
        :rtype: tuple
        """
        p = np.zeros((int(self._nclassifiers * (self._nclassifiers-1) / 2),),
                     dtype=np.float32)
        c = 0
        for i in range(self._nclassifiers - 1):
            for j in range(i+1, self._nclassifiers):
                p[c] = stats.wilcoxon(
                    self._accuracies[:, i],
                    self._accuracies[:, j],
                    zero_method='pratt',
                )[1]
                c += 1
        p_order = np.argsort(p)
        holm_bonferroni = alpha / np.arange(p.shape[0], 0, -1)
        p_significant = (p[p_order] <= holm_bonferroni)[p_order.argsort()]

        # calculate average ranks
        avg_ranks = stats.rankdata(self._accuracies, axis=1)
        avg_ranks = self._nclassifiers - avg_ranks + 1
        avg_ranks = avg_ranks.mean(axis=0)
        avg_ranks_order = avg_ranks.argsort()[::-1]

        lowest_rank = min(1, int(np.floor(avg_ranks.min())))
        highest_rank = max(len(avg_ranks), int(np.ceil(avg_ranks.max())))

        width = 6 + 0.3 * max(map(len, self._labels))
        height = 1.0 + self._nclassifiers * 0.1

        # initialize plot
        fig, ax = plt.subplots(1, 1, figsize=(width, height))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        ax.spines['right'].set_color("none")
        ax.spines['left'].set_color("none")
        ax.yaxis.set_major_locator(ticker.NullLocator())
        ax.spines['bottom'].set_color("none")
        ax.spines['top'].set_linewidth(2.5)
        ax.xaxis.set_ticks_position('top')
        ax.tick_params(which='major', width=2.5, length=5, labelsize=12)
        ax.tick_params(which='minor', width=2.0, length=3, labelsize=12)
        ax.set_xlim(highest_rank, lowest_rank)
        ax.set_ylim(0.0, 1.0)
        fig.subplots_adjust(bottom=-0.6, top=0.7)

        half = int(np.ceil(self._nclassifiers / 2))

        # visual configurations
        rank_xshift: float = 0.02 * (highest_rank-lowest_rank)
        label_xshift: float = 0.05 * (highest_rank-lowest_rank)
        label_offset: float = 0.01 * (highest_rank-lowest_rank)
        first_marking: float = 0.6
        markings_vspace: float = 0.35 * 1/half
        markings_color: tuple = (0.15, 0.15, 0.15, 1.0)
        cliques_color: tuple = _get_color(1) + (0.8,)

        # draw left branching markings
        for i, index in enumerate(avg_ranks_order[:half]):
            ax.axvline(
                x=avg_ranks[index],
                ymin=first_marking + (half-i-1)*markings_vspace,
                ymax=1.0,
                c=markings_color,
                lw=2.0,
            )
            ax.axhline(
                y=first_marking + (half-i-1)*markings_vspace,
                xmin=(half-i-1) * label_xshift / (highest_rank-lowest_rank),
                xmax=((highest_rank-avg_ranks[index])
                      / (highest_rank-lowest_rank)),
                c=markings_color,
                lw=2.0,
            )
            ax.text(highest_rank - rank_xshift - (half-i-1)*label_xshift,
                    first_marking + (half-i-1)*markings_vspace,
                    f"{avg_ranks[index]:.2f}",
                    ha="left", va="bottom", size=8)
            ax.text(highest_rank - (half-i-1)*label_xshift + label_offset,
                    first_marking + (half-i-1)*markings_vspace,
                    f"{self._labels[index]}",
                    ha="right", va="center", size=14)

        # draw right branching markings
        for i, index in enumerate(avg_ranks_order[half:]):
            ax.axvline(
                x=avg_ranks[index],
                ymin=first_marking + i*markings_vspace,
                ymax=1.0,
                c=markings_color,
                lw=2.0,
            )
            ax.axhline(
                y=first_marking + i*markings_vspace,
                xmin=((highest_rank-avg_ranks[index])
                      / (highest_rank-lowest_rank)),
                xmax=1.0 - i * label_xshift / (highest_rank-lowest_rank),
                c=markings_color,
                lw=2.0,
            )
            ax.text(
                lowest_rank + rank_xshift + i*label_xshift,
                first_marking + i*markings_vspace,
                f"{avg_ranks[index]:.2f}",
                ha="right", va="bottom", size=8,
            )
            ax.text(
                lowest_rank + i*label_xshift - label_offset,
                first_marking + i*markings_vspace,
                f"{self._labels[index]}",
                ha="left", va="center", size=14,
            )

        # get cliques based on the calculated p-values
        adjacency_matrix = np.zeros((self._nclassifiers, self._nclassifiers))
        connect_at = np.where(~p_significant)
        indexing = np.array(np.triu_indices(self._nclassifiers, k=1))
        for index in connect_at:
            i, j = indexing[:, index]
            adjacency_matrix[i, j] = 1
        ccliques = list(nx.find_cliques(nx.Graph(adjacency_matrix)))
        cliques = []
        for clique in ccliques:
            if len(clique) > 1:
                cliques.append(clique)

        # draw the cliques
        i = 1
        if len(cliques) < 4:
            first_clique_line = 0.9 + (len(cliques) + 4) / 100
        else:
            first_clique_line = 0.97
        clique_line_diff = (1 - (first_marking + (half-1)*markings_vspace))
        clique_line_diff -= 0.001
        if len(cliques) > 0:
            clique_line_diff /= len(cliques)
        clique_line_y = first_clique_line
        for clique in cliques:
            left = min(clique)
            right = max(clique)
            ax.axhline(
                y=clique_line_y,
                xmin=((highest_rank-avg_ranks[avg_ranks_order[left]])
                      / (highest_rank-lowest_rank)),
                xmax=((highest_rank-avg_ranks[avg_ranks_order[right]])
                      / (highest_rank-lowest_rank)),
                color=cliques_color,
                linewidth=4.0,
            )
            clique_line_y -= clique_line_diff

        return fig, ax


def _get_user_input():
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--csv_files", type=str,
                        help="CSV File names with accuracy results "
                             + "seperated by ';'",
                        required=True)
    parser.add_argument("-p", "--file_path", type=str,
                        help="Default path for the csv files",
                        default="")

    parser.add_argument("-c", "--columns", type=str,
                        help="Names of columns in the given files with the "
                             + " data that is going to be compared",
                        default=None)
    parser.add_argument("-l", "--labels", type=str,
                        help="Labels for the different methods that "
                             + "are compared seperated by ';'",
                        default=None)
    parser.add_argument("-o", "--opacity_column",
                        help="Color in points based on this column",
                        default=None)

    parser.add_argument("-sp", "--scatter_plot",
                        help="Show the scatter plots",
                        action="store_true")
    parser.add_argument("-cd", "--critical_difference",
                        help="Show the critical difference diagram",
                        action="store_true")
    parser.add_argument("-s", "--save_figure",
                        help="Save a shown figure. "
                             + "Use this option together with '-cd' or '-sp'.",
                        action="store_true")
    parser.add_argument("-t", "--test",
                        help="Do a wilcoxon test for all paired methods",
                        action="store_true")
    parser.add_argument("-n", "--figure_name",
                        help="Name of the image file",
                        type=str, default=None)

    return parser.parse_args()


def main():
    args = _get_user_input()

    files = args.csv_files.split(";")
    labels = files
    files = list(map(lambda x: x if x.endswith(".csv") else x + ".csv", files))
    if args.file_path is not None:
        files = list(map(lambda x: os.path.join(args.file_path, x), files))

    columns = [DEFAULT_COMPARISION_COLUMN] * len(files)
    if args.columns is not None:
        columns = args.columns.split(";")
    if args.labels is not None:
        labels = args.labels.split(";")

    f = pd.read_csv(files[0])
    accs = np.zeros((len(f), len(files)))
    for i in range(len(files)):
        accs[:, i] = pd.read_csv(files[i])[columns[i]]

    opacity = args.opacity_column
    if opacity is not None:
        opacity = f[opacity]
    else:
        opacity = f["TrS"] + f["TeS"]

    comparision = ClassifierComparision(accs, labels)

    if args.test:
        print("\nOne-sided paired Wilcoxon signed-rank test")
        print("------------------------------------------")
        for i in range(len(files)):
            for j in range(len(files)):
                if i == j:
                    continue
                print(f"H0: {labels[i]} <= {labels[j]} "
                      + f"\t H1: {labels[i]} > {labels[j]}")
                T, p = comparision.test_greater(i, j)
                print(f"\n{T = }, {p = }")
                print("------------------------------------------")

    if args.scatter_plot:
        fig1, axs = comparision.scatterplot(opacity=opacity)

    if args.critical_difference:
        fig2, ax = comparision.critical_difference_diagram()

    if args.save_figure:
        name = "comparison"
        if args.figure_name is not None:
            name = args.figure_name
        if args.critical_difference:
            plt.savefig(f"{name}.jpg", dpi=256)
        elif args.scatter_plot:
            plt.savefig(f"{name}.jpg", dpi=512, bbox_inches="tight")

    if args.critical_difference or args.scatter_plot:
        plt.show()


if __name__ == '__main__':
    main()
