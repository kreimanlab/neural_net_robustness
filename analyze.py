import pickle
import argparse
import itertools
import os
from collections import OrderedDict

import numpy as np
from matplotlib import pyplot

from results import get_results_filepath
from weights import load_weights, walk, walk_key_chain


def __get_weights_diff(weights1, weights2):
    return walk(weights1, collect=lambda key_chain, w1: w1 - walk_key_chain(weights2, key_chain))


def __relativize(weights):
    return walk(weights, collect=lambda _, x: x / x.size)


def __absolute_sum(weights):
    return walk(weights, collect=lambda _, x: np.absolute(x).sum())


def __plot_bar(x, y, xticks=None, ylabel=None, save_filepath=None, **kwargs):
    pyplot.bar(x, y, **kwargs)
    pyplot.ylabel(ylabel)
    if xticks is not None:
        pyplot.xticks(x, xticks, rotation=90, ha='left')
    if save_filepath is not None:
        os.makedirs(os.path.dirname(save_filepath), exist_ok=True)
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


def __plot_num_weights(weights):
    for weights_name, weights_values in weights.items():
        num_weights = walk(weights_values, collect=lambda _, x: x.size)
        __plot_weight_metric(num_weights, "figures/num_weights/%s.pdf" % weights_name)


def __plot_weights_diffs(weights):
    assert len(weights) >= 2, "Need at least two weights to compare"
    # find pairs
    combinations = map(OrderedDict, itertools.combinations(weights.items(), 2))
    for w1_name, w2_name in combinations:
        # compute differences
        weights_diffs = __get_weights_diff(weights[w1_name], weights[w2_name])
        relative_total_diffs = __absolute_sum(__relativize(weights_diffs))
        # plot
        __plot_weight_metric(relative_total_diffs,
                             "figures/weight_diffs/%s-vs-%s.pdf" % (w1_name, w2_name),
                             ylabel='Relative absolute diffs')


def __get_results(dataset, weights_name):
    results_filepaths = get_results_filepath(dataset, weights_name, search_variations=True)
    metrics = []
    for filepath in results_filepaths:
        with open(filepath, 'rb') as results_file:
            results = pickle.load(results_file)
        metrics.append(results['metric'])
    return np.mean(metrics), np.std(metrics)


def __plot_performances_by_weights(weights, datasets):
    for weights_name in weights:
        metric_means = []
        metric_errs = []
        for dataset in datasets:
            mean, err = __get_results(dataset, weights_name)
            metric_means.append(mean)
            metric_errs.append(err)
        __plot_bar(range(len(metric_means)), metric_means,
                   save_filepath="figures/performance_by_weights/%s.pdf" % weights_name,
                   yerr=metric_errs)


def __plot_performances_by_datasets(weights, datasets):
    for dataset in datasets:
        metric_means = []
        metric_errs = []
        for weights_name in weights:
            mean, err = __get_results(dataset, weights_name)
            metric_means.append(mean)
            metric_errs.append(err)
        __plot_bar(range(len(metric_means)), metric_means,
                   save_filepath="figures/performance_by_dataset/%s.pdf" % dataset,
                   yerr=metric_errs)


def __plot_performances(weights, datasets):
    __plot_performances_by_weights(weights, datasets)
    __plot_performances_by_datasets(weights, datasets)


if __name__ == '__main__':
    # options
    tasks = {'num_weights': lambda weights, _: __plot_num_weights(weights),
             'weight_diffs': lambda weights, _: __plot_weights_diffs(weights),
             'performance': __plot_performances}
    # params - command line
    parser = argparse.ArgumentParser(description='Neural Net Robustness - Analysis')
    parser.add_argument('task', type=str, choices=tasks.keys())
    parser.add_argument('--weights', type=str, nargs='+', default=["alexnet"],
                        help='The set of weights to compare with each other')
    parser.add_argument('--datasets', type=str, nargs='+', default=['ILSVRC2012/test'],
                        help='The datasets to compare the evaluations on')
    args = parser.parse_args()
    print('Running analysis with args', args)
    weights = args.weights if args.task == 'performance' else load_weights(*args.weights, keep_names=True)
    task = tasks[args.task]
    task(weights, args.datasets)
