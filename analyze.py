import argparse
import os
import h5py
import numpy as np
from functools import reduce
from matplotlib import pyplot


def walk(dictionary, collect, key_chain=None):
    result = {}
    for key, item in dictionary.items():
        sub_key_chain = (key_chain if key_chain is not None else []) + [key]
        if callable(getattr(item, "items", None)):
            result[key] = walk(item, collect, key_chain=sub_key_chain)
        else:
            result[key] = collect(sub_key_chain, item)
    return result


def walk_key_chain(dictionary, key_chain):
    """
    Walks down the nesting structure of a dictionary, following the keys in the `key_chain`.

    Example:
        d = {'a':
              {'b':
                {'c': 15}
              }
            }
        walk_key_chain(d, ['a', 'b', 'c'])  # returns 15
    :param dictionary: a nested dictionary containing other dictionaries
    :param key_chain: a list of keys to traverse down the nesting structure
    :return: the value in the nested structure after traversing down the `key_chain`
    """
    return reduce(lambda d, k: d[k], key_chain, dictionary)


def load_weights(filepath):
    with h5py.File(filepath, 'r') as file:
        weights = walk(file, lambda _, x: np.array(x))
    return weights


def get_weights_diff(weights1, weights2):
    return walk(weights1, collect=lambda key_chain, w1: w1 - walk_key_chain(weights2, key_chain))


def sum_abs_weights(weights):
    return walk(weights, collect=lambda _, x: np.absolute(x).sum())


def plot_weight_diffs(weight_diffs, figure_filename):
    keys = []
    values = []

    def collect_values(key_chain, value):
        keys.append(key_chain[-1])
        values.append(value)

    walk(weight_diffs, collect_values)
    keys, values = tuple(zip(*sorted(zip(keys, values))))
    key_indices = range(len(keys))
    pyplot.bar(key_indices, values)
    pyplot.xticks(key_indices, keys, rotation=90, ha='left')
    pyplot.savefig(figure_filename, bbox_inches='tight')
    pyplot.close()


if __name__ == '__main__':
    # params - command line
    parser = argparse.ArgumentParser(description='Neural Net Robustness - Weights Analysis')
    parser.add_argument('--weights1', type=str, default="alexnet",
                        help='The first set of weights to compare with weights2')
    parser.add_argument('--weights2', type=str, default="alexnet_retrained_on_VOC2012_for_10_epochs",
                        help='The second set of weights to compare with weights1')
    args = parser.parse_args()
    print('Running analysis with args', args)
    # compute differences
    weights1 = load_weights("weights/%s.h5" % args.weights1)
    weights2 = load_weights("weights/%s.h5" % args.weights2)
    layer_diffs = get_weights_diff(weights1, weights2)
    layer_diffs = sum_abs_weights(layer_diffs)
    # plot
    if not os.path.isdir("figures"):
        os.mkdir("figures")
    plot_weight_diffs(layer_diffs, "figures/%s-vs-%s.pdf" % (args.weights1, args.weights2))
