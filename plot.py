import argparse
import os
import pickle

import numpy as np
from matplotlib import pyplot

from results import get_results_filepath
from weights import load_weights, walk, walk_key_chain


def _get_weights_diff(weights1, weights2):
    return walk(weights1, collect=lambda key_chain, w1: w1 - walk_key_chain(weights2, key_chain))


def _relativize(weights, base_weights):
    return walk(weights, collect=lambda key_chain, x: x / np.absolute(walk_key_chain(base_weights, key_chain)).sum())


def _absolute(weights):
    return walk(weights, collect=lambda _, x: np.absolute(x))


def _means(weights):
    return walk(weights, collect=lambda _, x: np.mean(x))


def _stds(weights):
    return walk(weights, collect=lambda _, x: np.std(x))


def _sum(weights):
    return walk(weights, collect=lambda _, x: np.sum(x))


def _plot_bar(x, y, xticks=None, ylabel=None, save_filepath=None, **kwargs):
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


def _z_score(weight_values, weight_means, weight_stds):
    def z_score(key_chain, values):
        mean = walk_key_chain(weight_means, key_chain)
        assert np.isscalar(mean)
        std = walk_key_chain(weight_stds, key_chain)
        assert np.isscalar(std)
        return (values - mean) / std

    return walk(weight_values, collect=z_score)


def _plot_weight_metric(weights_metric, figure_filename, ylabel=None):
    keys = []
    values = []

    def collect_values(key_chain, value):
        keys.append(key_chain[-1])
        values.append(value)

    walk(weights_metric, collect_values)
    keys, values = tuple(zip(*sorted(zip(keys, values))))
    key_indices = range(len(keys))
    _plot_bar(key_indices, values, xticks=keys, ylabel=ylabel, save_filepath=figure_filename)


def _plot_num_weights(weights):
    for weights_name, weights_values in weights.items():
        num_weights = walk(weights_values, collect=lambda _, x: x.size)
        _plot_weight_metric(num_weights, "figures/num_weights/%s.pdf" % weights_name)


def _plot_weights_diffs(weights):
    assert len(weights) >= 2, "Need at least two weights to compare"
    # find pairs
    base_weight = next(weights.keys().__iter__())
    compare_weights = weights.keys() - [base_weight]
    for compare_name in compare_weights:
        # compute differences
        weights_diffs = _get_weights_diff(weights[base_weight], weights[compare_name])
        z_scored_diffs = _absolute(_z_score(_absolute(weights_diffs),
                                            _means(_absolute(weights[base_weight])),
                                            _stds(_absolute(weights[base_weight]))))
        z_scored_sums = _sum(z_scored_diffs)
        # plot
        _plot_weight_metric(z_scored_sums, "figures/weight_diffs/%s-vs-%s.pdf" % (base_weight, compare_name),
                            ylabel=r'z-scored diffs $\sum_{w=1}^{num weights^{layer}} '
                                   r'|\frac{|weights_w^{layer}| - mean(|weights^{layer}|}{std(|weights^{layer}|)}|$')


def _get_results(dataset, weights_name):
    results_filepaths, _ = get_results_filepath(dataset, weights_name, variations=True)
    metrics = []
    for filepath in results_filepaths:
        with open(filepath, 'rb') as results_file:
            results = pickle.load(results_file)
        metrics.append(results['results'])
    return metrics


def _append_metrics(metric_means, metric_errs, metrics, metric_name):
    if metric_name not in metric_means:
        metric_means[metric_name] = []
        metric_errs[metric_name] = []
    metrics = [metric[metric_name] for metric in metrics] if isinstance(metrics, list) else metrics[metric_name]
    mean, err = np.mean(metrics), np.std(metrics)
    metric_means[metric_name].append(mean)
    metric_errs[metric_name].append(err)


def _plot_performances_by_datasets(weights, datasets, metric_names):
    for dataset in datasets:
        metric_means = {}
        metric_errs = {}
        for weights_name in weights:
            metrics = _get_results(dataset, weights_name)
            for metric_name in metric_names:
                _append_metrics(metric_means, metric_errs, metrics, metric_name)
        for metric_name in metric_means:
            save_filepath = "figures/performance_by_dataset/%s-%s.pdf" % (dataset, metric_name)
            _plot_bar(range(1, len(metric_means[metric_name]) + 1), metric_means[metric_name],
                      save_filepath=save_filepath,
                      yerr=metric_errs[metric_name], xticks=[w for w in weights], ylabel=metric_name)


def main():
    # options
    tasks = {'num_weights': lambda weights, d, m: _plot_num_weights(weights),
             'weight_diffs': lambda weights, d, m: _plot_weights_diffs(weights),
             'performance': _plot_performances_by_datasets}
    # params - command line
    parser = argparse.ArgumentParser(description='Neural Net Robustness - Plot')
    parser.add_argument('task', type=str, choices=tasks.keys())
    parser.add_argument('--weights', type=str, nargs='+', default=["alexnet"],
                        help='The set of weights to compare with each other')
    parser.add_argument('--datasets', type=str, nargs='+', default=['ILSVRC2012/val'],
                        help='The datasets to compare the evaluations on')
    parser.add_argument('--metrics', type=str, nargs='+', default=['top5error', 'top1error'],
                        help='The metrics to use for performance')
    args = parser.parse_args()
    print('Running plot with args', args)
    weights = args.weights if args.task == 'performance' else load_weights(*args.weights, keep_names=True)
    task = tasks[args.task]
    task(weights, args.datasets, args.metrics)


if __name__ == '__main__':
    main()
