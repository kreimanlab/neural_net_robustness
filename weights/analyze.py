import numpy as np

from weights import walk, walk_key_chain


def weight_differences(weights1, weights2):
    return walk(weights1, collect=lambda key_chain, w1: w1 - walk_key_chain(weights2, key_chain))


def absolute(weights):
    return walk(weights, collect=lambda _, x: np.absolute(x))


def means(weights):
    return walk(weights, collect=lambda _, x: np.mean(x))


def stds(weights):
    return walk(weights, collect=lambda _, x: np.std(x))


def sum(weights):
    return walk(weights, collect=lambda _, x: np.sum(x))


def max(weights):
    max_value = float('-inf')

    def find_max(_, x):
        nonlocal max_value
        if x > max_value:
            max_value = x

    walk(weights, collect=find_max)
    return max_value


def count(weights):
    return walk(weights, collect=lambda _, x: x.size)


def divide(denominator, divider):
    return walk(denominator, collect=lambda key_chain, x: x / walk_key_chain(divider, key_chain))


def relativize(weights, base_weights):
    return walk(weights, collect=lambda key_chain, x: x / walk_key_chain(base_weights, key_chain))


def proportion_different(weights1, weights2, mean_across_layers=False):
    """
    Returns the number of weights that changed across all layers
    divided by the total number of weights across all layers.
    """
    assert weights1.keys() == weights2.keys()

    def collect_proportion_different(key_chain, w1):
        w2 = walk_key_chain(weights2, key_chain)
        assert w2.size == w1.size
        return (w1 != w2).sum() / w1.size

    proportions_per_layer = walk(weights1, collect=collect_proportion_different)
    if not mean_across_layers:
        return proportions_per_layer

    proportions = []

    def collect_proportions(_, proportion):
        nonlocal proportions
        proportions.append(proportion)

    walk(proportions_per_layer, collect=collect_proportions)
    return np.mean(proportions)


def z_score(weight_values, weight_means, weight_stds):
    def _z_score(key_chain, values):
        mean = walk_key_chain(weight_means, key_chain)
        assert np.isscalar(mean)
        std = walk_key_chain(weight_stds, key_chain)
        assert np.isscalar(std)
        return (values - mean) / std

    return walk(weight_values, collect=_z_score)


def summed_absolute_relative_diffs(weights, base_weights):
    weights_diffs = weight_differences(weights, base_weights)
    relativized_diffs = walk(weights_diffs, collect=lambda key_chain, diffs:
    diffs / walk_key_chain(base_weights, key_chain))
    return sum(absolute(relativized_diffs))


def relative_summed_absolute_diffs(weights, base_weights):
    weights_diffs = weight_differences(weights, base_weights)
    summed_absolute_diffs = sum(absolute(weights_diffs))
    summed_absolute_base = sum(absolute(base_weights))
    return walk(summed_absolute_diffs, collect=lambda key_chain, diffs:
    diffs / walk_key_chain(summed_absolute_base, key_chain))
