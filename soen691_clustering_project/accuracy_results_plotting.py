# -*- coding: utf-8 -*-

"""Plotting class."""
from matplotlib import pyplot as plt


def label_bars(rects, ax):
    # Get y-axis height to calculate label position from.
    (y_bottom, y_top) = ax.get_ylim()
    y_height = y_top - y_bottom

    for i, rect in enumerate(rects):
        height = rect.get_height()

        # Fraction of axis height taken up by this rectangle
        p_height = (height / y_height)

        # If we can fit the label above the column, do that;
        # otherwise, put it inside the column.
        if p_height > 0.975:  # arbitrary; 95% looked good to me.
            label_position = height - (y_height * 0.05)
        else:
            label_position = height + (y_height * 0.005)

        if i == 0:
            ax.text(rect.get_x() + rect.get_width() / 2., label_position,
                    '{:.2f}%'.format(float(height)),
                    ha='center', va='bottom')
        else:
            ax.text(rect.get_x() + rect.get_width() / 2., label_position,
                    '{:.2f}%'.format(float(height)),
                    ha='center', va='bottom')


def plot_bars(fig, bars, heights, title, ax=None):
    if not ax:
        ax = fig.add_subplot(111)

    bar_list = ax.bar(bars, heights)
    label_bars(bar_list, ax)

    bar_list[0].set_color('r')
    bar_list[2].set_color('g')
    bar_list[3].set_color('grey')

    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Clustering Algorithm')
    ax.set_yticklabels(['{:}%'.format(x) for x in ax.get_yticks()])
    ax.set_title(title, color='red')


if __name__ == '__main__':
    circles = [50.0, 52.28571428571429, 100.0, 50.142857142857146]
    spiral = [33.01282051282052, 35.89743589743589, 100.0, 33.65384615384615]
    jain = [78.55227882037534, 86.05898123324397, 100.0, 77.4798927613941]
    pathbased = [42.47491638795987, 42.47491638795987, 100.0, 43.14381270903011]

    algorithms = ['KMEANS', 'HAC', 'CURE', 'BFR']

    figure = plt.figure(figsize=(14, 8))
    figure.set_tight_layout(True)

    # ax_circles = figure.add_subplot(221)
    # ax_spiral = figure.add_subplot(222)
    # ax_jain = figure.add_subplot(223)
    # ax_pathbased = figure.add_subplot(224)

    # plot_bars(figure, algorithms, circles, 'Circles', ax=ax_circles)
    # plot_bars(figure, algorithms, spiral, 'Path-based2: SPIRAL', ax=ax_spiral)
    # plot_bars(figure, algorithms, jain, 'Jain', ax=ax_jain)
    # plot_bars(figure, algorithms, pathbased, 'Path-based1', ax=ax_pathbased)
    # plt.show()

    # plot_bars(figure, algorithms, circles, 'Circles')
    # plt.show()
    # plot_bars(figure, algorithms, spiral, 'Path-based2: SPIRAL')
    # plt.show()
    # plot_bars(figure, algorithms, jain, 'Jain')
    # plt.show()
    # plot_bars(figure, algorithms, pathbased, 'Path-based1')
    # plt.show()


    atom = [30.874999999999996, 47.5, 100.0, 30.874999999999996]
    chain_link = [65.4, 81.80000000000001, 100.0, 64.8]

    algorithms = ['KMEANS', 'HAC', 'CURE', 'BFR']

    figure = plt.figure(figsize=(14, 8))
    figure.set_tight_layout(True)

    # ax_atom = figure.add_subplot(121)
    # ax_chain_link = figure.add_subplot(122)
    #
    # plot_bars(figure, algorithms, atom, 'FCPS Atom', ax=ax_atom)
    # plot_bars(figure, algorithms, chain_link, 'FCPS Chainlink', ax=ax_chain_link)
    #
    # plt.show()

    # plot_bars(figure, algorithms, atom, 'FCPS Atom')
    # plt.show()
    plot_bars(figure, algorithms, chain_link, 'FCPS Chainlink')
    plt.show()

