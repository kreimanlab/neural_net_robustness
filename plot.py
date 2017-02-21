import argparse
import os
import pickle
import re
from collections import defaultdict
import seaborn as sns

import numpy as np
import matplotlib
from matplotlib import pyplot

from results import get_results_filepath
<<<<<<< HEAD
from weights import load_weights, walk, has_sub_layers, merge_sub_layers
from weights.analyze import weight_differences, absolute, means, medians, stds, sum, max, \
=======
from weights import load_weights, walk, has_sub_layers, merge_sub_layers, expand_weights_names
from weights.analyze import weight_differences, absolute, means, stds, sum, max, \
>>>>>>> fc3a8997aab786c3d6ca7c0dbdd42b2983f30135
    z_score, summed_absolute_relative_diffs, count, divide

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rc('xtick', labelsize=16) 
# matplotlib.rc('ytick', labelsize=20) 
matplotlib.rc('axes', labelsize=14) 
matplotlib.rc('ytick', labelsize=14) 
matplotlib.rc('xtick', labelsize=14) 

colorList = sns.color_palette()+sns.color_palette("husl", 8)[:2]


def _plot_bar(x, y, xticks=None, ylabel=None, save_filepath=None, **kwargs):
    # color = [sns.color_palette()[1] for _ in range(len(xticks))]
    color = []
    for tick in xticks:
        if "conv" in tick:
            color.append(colorList[int(tick.split("conv_")[1].split("_")[0])-1])
        else:
            color.append(colorList[int(tick.split("dense_")[1].split("_")[0])+4])
    matplotlib.rc('xtick', labelsize=16) 

    pyplot.bar(x, y, color=color, **kwargs)
    pyplot.ylabel(ylabel)
    ax = pyplot.gca()
    ax.yaxis.set_label_coords(-0.05,0.35)
    if xticks is not None:
        pyplot.xticks(x, xticks, rotation=90, ha='left')
        ax.set_xlim([0,len(xticks)])
    if save_filepath is not None:
        os.makedirs(os.path.dirname(save_filepath), exist_ok=True)
        pyplot.tight_layout()
        pyplot.savefig(save_filepath, bbox_inches='tight')
        pyplot.close()
    else:
        pyplot.show()


def _sorted_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    if not handles:  # no perturbations
        return
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels, loc='lower left')


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
        num_weights = count(weights_values)
        _plot_weight_metric(num_weights, "figures/num_weights/%s.pdf" % weights_name)


def _summed_absolute_zscore(weights, base_weights, weights_name, base_weights_name):
    weights_diffs = weight_differences(base_weights, weights)
    z_scored_diffs = absolute(z_score(absolute(weights_diffs),
                                      means(absolute(base_weights)),
                                      stds(absolute(base_weights))))
    summed_z_scores = means(z_scored_diffs)
    # summed_z_scores = medians(z_scored_diffs)
    # summed_z_scores = sum(z_scored_diffs)

    _plot_weight_metric(summed_z_scores, "figures/weight_diffs/%s-vs-%s-zscore.pdf" % (
        base_weights_name, weights_name.replace('/', '_')),
                        ylabel='mean absolute z-score of weight change from retraining')


def _relativized_value_diffs(weights, base_weights, weights_name, base_weights_name):
    relativized_absolute_summed_diffs = summed_absolute_relative_diffs(weights, base_weights)
    _plot_weight_metric(relativized_absolute_summed_diffs, "figures/weight_diffs/%s-vs-%s-diff_by_basesum.pdf" % (
        base_weights_name, weights_name.replace('/', '_')),
                        ylabel='differences relativized to sum of base weights')


def _relativized_value_and_num_diffs(weights, base_weights, weights_name, base_weights_name):
    relativized_absolute_summed_diffs = summed_absolute_relative_diffs(weights, base_weights)
    num_weights = count(base_weights)
    relativized_to_num_weights = divide(relativized_absolute_summed_diffs, num_weights)
    _plot_weight_metric(relativized_to_num_weights, "figures/weight_diffs/%s-vs-%s-diff_by_basesumandnum.pdf" % (
        base_weights_name, weights_name.replace('/', '_')),
                        ylabel='differences relativized to sum and number of base weights')


def _hist(weights, base_weights, weights_name, base_weights_name):
    for layer, layer_weights in weights.items():
        if not layer_weights and not has_sub_layers(weights, layer):
            continue
        if has_sub_layers(weights, layer):
            layer_weights = merge_sub_layers(weights, layer)
            base_layer_weights = merge_sub_layers(base_weights, layer)
        else:
            base_layer_weights = base_weights[layer]
        layer_weights = np.concatenate([v.flatten() for v in layer_weights.values()])
        base_layer_weights = np.concatenate([v.flatten() for v in base_layer_weights.values()])
        assert base_layer_weights.shape == layer_weights.shape

        bins = min(100, int(np.ceil(np.sqrt(layer_weights.size))))
        fig, ax = pyplot.subplots()
        ax.set_yscale('log')
        ax.hist(base_layer_weights.flatten(), bins, alpha=0.5, label=base_weights_name)
        ax.hist(layer_weights.flatten(), bins, alpha=0.5, label=weights_name)
        ax.set_ylim(ax.get_ylim() * np.array([1, 10]))
        ax.legend()
        save_filepath = "figures/weight_hists/%s-vs-%s--%s.pdf" % (
            base_weights_name, weights_name.replace('/', '_'), layer)
        pyplot.savefig(save_filepath, bbox_inches='tight')
        pyplot.close(fig)


def _plot_weights_diffs(weights):
    assert len(weights) >= 2, "Need at least two weights to compare"
    # find pairs
    base_weight = next(weights.keys().__iter__())
    compare_weights = weights.keys() - [base_weight]
    for compare_name in compare_weights:
        # compute differences
        # for metric in [_relativized_value_and_num_diffs]:
        for metric in [_summed_absolute_zscore]:
            metric(weights[compare_name], weights[base_weight], compare_name, base_weight)


def _get_results(dataset, weights_name):
    results_filepaths, _ = get_results_filepath(dataset, weights_name, variations=True)
    assert results_filepaths, "no result files found for dataset %s and weights %s" % (dataset, weights_name)
    metrics = defaultdict(list)
    for filepath in results_filepaths:
        with open(filepath, 'rb') as results_file:
            results = pickle.load(results_file)
        for key in results:
            if type(results[key]) is not dict:
                metrics[key].append(results[key])
            else:
                if key not in metrics:
                    metrics[key] = defaultdict(list)
                for subkey in results[key]:
                    metrics[key][subkey].append(results[key][subkey])
    assert metrics, "no metrics found in %s" % ",".join(results_filepaths)
    return metrics


def _get_weights_configuration(weights_name):
    if weights_name.count('-') == 2:  # layer perturbation
        model, layer, perturbation = weights_name.split('-')
        proportion_start = re.search("\d", perturbation).start()
        perturbation_type = perturbation[:proportion_start]
        perturbation_proportion = float(perturbation[proportion_start:])
    else:  # no perturbation
        model, layer, perturbation_type, perturbation_proportion = weights_name, None, None, 0
    return model, layer, perturbation_type, perturbation_proportion


def _get_weights_perturbation(weight_names):
    perturbation = set([perturbation for weight_name in weight_names
                        for _, _, perturbation, _ in [_get_weights_configuration(weight_name)]
                        if perturbation is not None])
    if not perturbation:  # no perturbation
        return ''
    assert len(perturbation) is 1
    return perturbation.pop()


def _collect_layer_performances(dataset, weight_names, metric_name):
    layer_metrics = defaultdict(lambda: defaultdict(list))
    for weights_name in weight_names:
        model, layer, perturbation, proportion = _get_weights_configuration(weights_name)
        results = _get_results(dataset, weights_name)
        metrics = results[metric_name]
        layer_metrics[layer][proportion] += metrics

    layer_means = defaultdict(dict)
    layer_errs = defaultdict(dict)
    for layer in layer_metrics:
        for proportion in layer_metrics[layer]:
            layer_means[layer][proportion] = np.mean(layer_metrics[layer][proportion])
            layer_errs[layer][proportion] = np.std(layer_metrics[layer][proportion])
    return layer_means, layer_errs


def _plot_performances_by_datasets(weight_names, datasets, metric_names):
    perturbation = _get_weights_perturbation(weight_names)
    for dataset in datasets:
        for metric_name in metric_names:
            layer_means, layer_errs = _collect_layer_performances(dataset, weight_names, metric_name)
            fig, ax = pyplot.subplots()
            ax.set_xlabel('weight mutations in multiples of variance')
            ax.set_ylabel(metric_name)
            for layer in layer_means:
                x = list(layer_means[layer].keys())
                y = list(layer_means[layer].values())
                err = list(layer_errs[layer].values())
                x, y, err = zip(*[(x_, y_, e_) for (x_, y_, e_) in sorted(zip(x, y, err))])
                if len(x) > 1:  # multiple measurements
                    ax.errorbar(x, y, yerr=err, label=layer)
                else:  # single measurement
                    ax.errorbar(x, y, yerr=err, label=layer, marker='o')
            ax.set_xlim(np.array(ax.get_xlim()) + np.array([-.25, .25]))
            ax.set_ylim(0, 1)
            # _sorted_legend(ax)
            save_filepath = "figures/performance_by_dataset/%s-%s-%s.pdf" % (dataset, perturbation, metric_name)
            print('Saving to %s...' % save_filepath)
            fig.savefig(save_filepath, bbox_inches='tight')
            pyplot.close(fig)


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
    parser.add_argument('--metrics', type=str, nargs='+',
                        default=['top5error', 'top1error', 'top5performance', 'top1performance'],
                        help='The metrics to use for performance')
    parser.add_argument('--weights_directory', type=str, default='weights',
                        help='The directory in which the weights are stored in')
    args = parser.parse_args()
    print('Running plot with args', args)
    weights = expand_weights_names(*args.weights) if args.task == 'performance' \
        else load_weights(*args.weights, keep_names=True, weights_directory=args.weights_directory)
    task = tasks[args.task]
    task(weights, args.datasets, args.metrics)


if __name__ == '__main__':
    main()
