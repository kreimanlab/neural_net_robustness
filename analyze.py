import argparse
import os
import pickle
import sys

import numpy as np

from datasets import get_data, validate_datasets
from predictions import get_predictions_filepath
from results import get_results_filepath
from weights import load_weights, walk, get_weights_filepath
from weights.analyze import proportion_different, relative_summed_absolute_diffs


def _argmax_n(a, n):
    ind = np.argpartition(a, -n)[-n:]
    ind = ind[np.argsort(a[ind])]
    return ind


def _top5(softmax):
    # TODO: we need to evaluate this with a SVM to get rid of errors due to re-ordering of labels
    return np.array([_argmax_n(s, 5) for s in softmax])


def _top1(softmax):
    return np.array([np.argmax(s) for s in softmax])


def performance(actual, predicted):
    assert len(predicted) == len(actual)
    num_correct = 0
    try:
        for real, predict in zip(actual, predicted):
            num_correct += any(p == real for p in predict)
    except TypeError:  # not iterable
        num_correct = sum(real == predict for real, predict in zip(actual, predicted))
    return num_correct / len(actual)


def error(actual, predicted):
    return 1 - performance(actual, predicted)


def _get_predictions(dataset, weights_name, predictions_directory):
    results_filepaths, nums = get_predictions_filepath(dataset, weights_name,
                                                       variations=True, predictions_directory=predictions_directory)
    predictions, nums_dict = {}, {}
    for filepath, num in zip(results_filepaths, nums):
        with open(filepath, 'rb') as results_file:
            results = pickle.load(results_file)
        predictions[filepath] = results['predictions']
        nums_dict[filepath] = num
    return predictions, nums_dict


def _evaluate_metric(metric, predictions, truths_mapping):
    truths, predictions = zip(*[(truths_mapping[image_path], prediction)
                                for image_path, prediction in predictions.items()])
    return metric(truths, predictions)


def _layer_proportion_different(base_weights, weights):
    """
    Find layer where the proportion difference is > 0 and return that.
    """
    proportions_different = proportion_different(base_weights, weights)
    nonzero_layer_names = []
    layer_proportions = []  # collect two proportions: one for the weights, one for the biases

    def collect_proportion(key_chain, p):
        if p == 0:
            return
        nonlocal nonzero_layer_names, layer_proportions
        nonzero_layer_names.append(key_chain[0])
        layer_proportions.append(p)

    walk(proportions_different, collect=collect_proportion)
    if len(layer_proportions) == 0:  # nothing changed
        return 0
    # make sure that proportions come from the same layer
    # and weights and biases as well as separate streams have similar proportions
    assert any(char.isdigit() for char in os.path.commonprefix(nonzero_layer_names)), \
        'weight differences come from different layers: ' + ", ".join(nonzero_layer_names)  # layer number present
    assert len(layer_proportions) % 2 == 0
    assert np.std(layer_proportions) < 0.1
    return np.mean(layer_proportions)


def _analyze(weight_names, datasets_directory, datasets, metrics, base_weights=None,
             weights_directory='weights', predictions_directory='predictions'):
    for dataset in datasets:
        truths_mapping = get_data(dataset, datasets_directory)
        for weights_name in weight_names:
            nummed_predictions, nums = _get_predictions(dataset, weights_name, predictions_directory)
            for prediction_filepath, prediction in nummed_predictions.items():
                print("Analyzing %s..." % prediction_filepath, end='')
                output = {'dataset': dataset, 'weights': weights_name}
                variation = nums[prediction_filepath] if nums[prediction_filepath] is not None else False
                for metric_name, metric in metrics.items():
                    print(" %s" % metric_name, end='')
                    sys.stdout.flush()
                    output[metric_name] = _evaluate_metric(metric, prediction, truths_mapping)
                print()
                if base_weights is not None:
                    weights_file = get_weights_filepath(weights_name, variations=variation,
                                                        weights_directory=weights_directory)
                    weight_values = load_weights(weights_file, weights_directory=weights_directory)
                    for base_weights_name, base_weights_values in base_weights.items():
                        print("Comparing weights '%s' with '%s'" % (weights_file, base_weights_name))
                        output['base_weights'] = base_weights_name
                        output['perturbation_proportion'] = _layer_proportion_different(base_weights_values,
                                                                                        weight_values)
                        output['relative_summed_absolute_weight_differences'] = \
                            relative_summed_absolute_diffs(weight_values, base_weights_values)

                results_filepath = get_results_filepath(dataset, weights_name, variations=variation)
                print("Writing results to %s" % results_filepath)
                pickle.dump(output, open(results_filepath, 'wb'))


def main():
    # options
    metrics = {'top5error': lambda actual, predicted: error(actual, _top5(predicted)),
               'top1error': lambda actual, predicted: error(actual, _top1(predicted)),
               'top5performance': lambda actual, predicted: performance(actual, _top5(predicted)),
               'top1performance': lambda actual, predicted: performance(actual, _top1(predicted))
               }
    # params - command line
    parser = argparse.ArgumentParser(description='Neural Net Robustness - Analysis')
    parser.add_argument('--weights', type=str, nargs='+', default=["alexnet"],
                        help='The set of weights to compare with each other')
    parser.add_argument('--datasets', type=str, nargs='+', default=['ILSVRC2012/val'],
                        help='The datasets to compare the evaluations on')
    parser.add_argument('--metrics', type=str, nargs='+', default=metrics.keys(),
                        choices=metrics.keys(), help='The metrics to use for performance')
    parser.add_argument('--base_weights', type=str, default=None,
                        help='The weights to compare perturbations with')
    parser.add_argument('--weights_directory', type=str, default='weights',
                        help='The directory all weights are stored in')
    parser.add_argument('--datasets_directory', type=str, default='datasets',
                        help='The directory all datasets are stored in')
    parser.add_argument('--predictions_directory', type=str, default='predictions',
                        help='The directory all predictions are stored in')
    args = parser.parse_args()
    print('Running analysis with args', args)
    validate_datasets(args.datasets, args.datasets_directory)
    base_weights = load_weights(args.base_weights, weights_directory=args.weights_directory, keep_names=True) \
        if args.base_weights is not None else None
    metrics = dict((metric, metrics[metric]) for metric in args.metrics)
    _analyze(args.weights, args.datasets_directory, args.datasets, metrics, base_weights,
             weights_directory=args.weights_directory, predictions_directory=args.predictions_directory)


if __name__ == '__main__':
    main()
