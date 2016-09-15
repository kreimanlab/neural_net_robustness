import argparse
import os
import h5py
import numpy as np
from functools import reduce
from matplotlib import pyplot


def __walk(dictionary, collect, key_chain=None):
    result = {}
    for key, item in dictionary.items():
        sub_key_chain = (key_chain if key_chain is not None else []) + [key]
        if callable(getattr(item, "items", None)):
            result[key] = __walk(item, collect, key_chain=sub_key_chain)
        else:
            result[key] = collect(sub_key_chain, item)
    return result


def __walk_key_chain(dictionary, key_chain):
    """
    Walks down the nesting structure of a dictionary, following the keys in the `key_chain`.

    Example:
        d = {'a':
              {'b':
                {'c': 15}
              }
            }
        __walk_key_chain(d, ['a', 'b', 'c'])  # returns 15
    :param dictionary: a nested dictionary containing other dictionaries
    :param key_chain: a list of keys to traverse down the nesting structure
    :return: the value in the nested structure after traversing down the `key_chain`
    """
    return reduce(lambda d, k: d[k], key_chain, dictionary)


def __load_weights(*weights_names):
    weights = list()
    for weights_name in weights_names:
        filepath = "weights/%s.h5" % weights_name
        with h5py.File(filepath, 'r') as file:
            w = __walk(file, lambda _, x: np.array(x))
            weights.append(w)
    return weights if len(weights) > 1 else weights[0]


def __get_weights_diff(weights1, weights2):
    return __walk(weights1, collect=lambda key_chain, w1: w1 - __walk_key_chain(weights2, key_chain))


def __relativize(weights):
    return __walk(weights, collect=lambda _, x: x / x.size)


def __absolute_sum(weights):
    return __walk(weights, collect=lambda _, x: np.absolute(x).sum())


def __plot_weight_metric(weights_metric, figure_filename, ylabel=None):
    keys = []
    values = []

    def collect_values(key_chain, value):
        keys.append(key_chain[-1])
        values.append(value)

    __walk(weights_metric, collect_values)
    keys, values = tuple(zip(*sorted(zip(keys, values))))
    key_indices = range(len(keys))
    pyplot.bar(key_indices, values)
    pyplot.ylabel(ylabel)
    pyplot.xticks(key_indices, keys, rotation=90, ha='left')
    pyplot.savefig(figure_filename, bbox_inches='tight')
    pyplot.close()


def __plot_num_weights(weights, weights_name):
    num_weights = __walk(weights, collect=lambda _, x: x.size)
    __plot_weight_metric(num_weights, "figures/%s-num_weights.pdf" % weights_name)


def __plot_weights_diffs(weights1, weights2, weights1_name, weights2_name):
    # compute differences
    weights_diffs = __get_weights_diff(weights1, weights2)
    relative_total_diffs = __absolute_sum(__relativize(weights_diffs))
    # plot
    __plot_weight_metric(relative_total_diffs,
                         "figures/%s-vs-%s.pdf" % (weights1_name, weights2_name),
                         ylabel='Relative absolute diffs')


if __name__ == '__main__':
    # params - command line
    parser = argparse.ArgumentParser(description='Neural Net Robustness - Analysis')
    parser.add_argument('--weights1', type=str, default="alexnet",
                        help='The first set of weights to compare with weights2')
    parser.add_argument('--weights2', type=str, default="alexnet_retrained_on_VOC2012_for_10_epochs",
                        help='The second set of weights to compare with weights1')
    args = parser.parse_args()
    print('Running analysis with args', args)
    weights1, weights2 = __load_weights(args.weights1, args.weights2)
    __plot_num_weights(weights1, args.weights1)
    __plot_weights_diffs(weights1, weights2, args.weights1, args.weights2)
