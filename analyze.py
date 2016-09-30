import os
import argparse
import pickle

import numpy as np
import sys

from datasets import get_data, validate_datasets
from predictions import get_predictions_filepath
from results import get_results_filepath


def __argmax_n(a, n):
    ind = np.argpartition(a, -n)[-n:]
    ind = ind[np.argsort(a[ind])]
    return ind


def __top5(softmax):
    # TODO: we need to evaluate this with a SVM to get rid of errors due to re-ordering of labels
    return np.array([__argmax_n(s, 5) for s in softmax])


def __top1(softmax):
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


def __get_predictions(dataset, weights_name):
    results_filepaths, nums = get_predictions_filepath(dataset, weights_name, variations=True)
    predictions, nums_dict = {}, {}
    for filepath, num in zip(results_filepaths, nums):
        with open(filepath, 'rb') as results_file:
            results = pickle.load(results_file)
        predictions[filepath] = results['predictions']
        nums_dict[filepath] = num
    return predictions, nums_dict


def __evaluate_metric(metric, predictions, truths_mapping):
    truths, predictions = zip(*[(truths_mapping[image_path], prediction)
                                for image_path, prediction in predictions.items()])
    return metric(truths, predictions)


def __analyze(weights, datasets_directory, datasets, metrics):
    for dataset in datasets:
        truths_mapping = get_data(dataset, datasets_directory)
        for weights_name in weights:
            nummed_predictions, nums = __get_predictions(dataset, weights_name)
            for prediction_filepath, prediction in nummed_predictions.items():
                print("Analyzing %s..." % prediction_filepath, end='')
                results = {}
                for metric_name, metric in metrics.items():
                    print(" %s" % metric_name, end='')
                    sys.stdout.flush()
                    results[metric_name] = __evaluate_metric(metric, prediction, truths_mapping)
                print()

                results_filepath = get_results_filepath(dataset, weights_name, nums[prediction_filepath])
                if os.path.isfile(results_filepath):
                    print("Merging with previous results")
                    previous_results = pickle.load(open(results_filepath, 'rb'))
                    merged_results = previous_results['results']
                    merged_results.update(results)
                    results = merged_results
                print("Writing results to %s" % results_filepath)
                pickle.dump({'results': results, 'dataset': dataset, 'weights': weights_name},
                            open(results_filepath, 'wb'))


if __name__ == '__main__':
    # options
    metrics = {'top5error': lambda actual, predicted: error(actual, __top5(predicted)),
               'top1error': lambda actual, predicted: error(actual, __top1(predicted))}
    # params - command line
    parser = argparse.ArgumentParser(description='Neural Net Robustness - Analysis')
    parser.add_argument('--weights', type=str, nargs='+', default=["alexnet"],
                        help='The set of weights to compare with each other')
    parser.add_argument('--datasets_directory', type=str, default='datasets',
                        help='The directory all datasets are stored in')
    parser.add_argument('--datasets', type=str, nargs='+', default=['ILSVRC2012/val'],
                        help='The datasets to compare the evaluations on')
    parser.add_argument('--metrics', type=str, nargs='+', default=[m for m in metrics],
                        choices=metrics.keys(), help='The metrics to use for performance')
    args = parser.parse_args()
    print('Running analysis with args', args)
    validate_datasets(args.datasets, args.datasets_directory)
    metrics = dict((metric, metrics[metric]) for metric in args.metrics)
    __analyze(args.weights, args.datasets_directory, args.datasets, metrics)
