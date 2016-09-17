import pickle
import argparse
import itertools
import os
from collections import OrderedDict

import numpy as np
from matplotlib import pyplot

from weights import load_weights, walk, walk_key_chain


def __get_weights_diff(weights1, weights2):
    return walk(weights1, collect=lambda key_chain, w1: w1 - walk_key_chain(weights2, key_chain))


def __relativize(weights):
    return walk(weights, collect=lambda _, x: x / x.size)


def __absolute_sum(weights):
    return walk(weights, collect=lambda _, x: np.absolute(x).sum())


def __plot_bar(x, y, xticks=None, ylabel=None, save_filepath=None):
    pyplot.bar(x, y)
    pyplot.ylabel(ylabel)
    if xticks is not None:
        pyplot.xticks(x, xticks, rotation=90, ha='left')
    if save_filepath is not None:
        pyplot.savefig(save_filepath, bbox_inches='tight')
        pyplot.close()
    else:
        pyplot.show()


def __plot_weight_metric(weights_metric, figure_filename, ylabel=None):
    keys = []
    values = []

    def collect_values(key_chain, value):
        keys.append(key_chain[-1])
        values.append(value)

    walk(weights_metric, collect_values)
    keys, values = tuple(zip(*sorted(zip(keys, values))))
    key_indices = range(len(keys))
    __plot_bar(key_indices, values, xticks=keys, ylabel=ylabel, save_filepath=figure_filename)


def __plot_num_weights(weights, weights_name):
    num_weights = walk(weights, collect=lambda _, x: x.size)
    __plot_weight_metric(num_weights, "figures/weights/%s-num_weights.pdf" % weights_name)


def __plot_weights_diffs(weights):
    # find pairs
    combinations = map(OrderedDict, itertools.combinations(weights.items(), 2))
    for w1_name, w2_name in combinations:
        # compute differences
        weights_diffs = __get_weights_diff(weights[w1_name], weights[w2_name])
        relative_total_diffs = __absolute_sum(__relativize(weights_diffs))
        # plot
        __plot_weight_metric(relative_total_diffs,
                             "figures/weights/%s-vs-%s.pdf" % (w1_name, w2_name),
                             ylabel='Relative absolute diffs')


def __plot_performances(weights, datasets):
    for weights_name in weights:
        metrics = []
        for dataset in datasets:
            results_filepath = "results/%s-%s.p" % (dataset, weights_name)
            with open(results_filepath, 'rb') as results_file:
                results = pickle.load(results_file)
            metrics.append(results['metric'])
        __plot_bar(range(len(metrics)), metrics, save_filepath="figures/performance/%s.pdf" % weights_name)


if __name__ == '__main__':
    # params - command line
    parser = argparse.ArgumentParser(description='Neural Net Robustness - Analysis')
    parser.add_argument('--weights', type=str, nargs='+',
                        default=["alexnet", "alexnet_retrained_on_VOC2012_for_10_epochs"],
                        help='The set of weights to compare with each other')
    parser.add_argument('--datasets', type=str, nargs='+', default=['VOC2012'],
                        help='The datasets to compare the evaluations on',
                        choices=[d for d in os.listdir("datasets") if os.path.isdir(os.path.join("datasets", d))])
    args = parser.parse_args()
    print('Running analysis with args', args)
    assert len(args.weights) >= 2, "Need at least two weights to compare"
    weights = load_weights(*args.weights, keep_names=True)
    # compare weights
    __plot_num_weights(weights[args.weights[0]], args.weights[0])
    __plot_weights_diffs(weights)
    # compare performance
    __plot_performances(weights, args.datasets)
