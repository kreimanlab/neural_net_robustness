import os
from collections import OrderedDict
from functools import reduce

import h5py
import numpy as np


def load_weights(*weights_names, keep_names=False):
    weights = list()
    for weights_name in weights_names:
        filepath = "weights/%s.h5" % weights_name
        with h5py.File(filepath, 'r') as file:
            w = walk(file, lambda _, x: np.array(x))
            weights.append(w)
    if not keep_names:
        return weights if len(weights) > 1 else weights[0]
    else:
        return OrderedDict(zip(weights_names, weights))


def walk(dictionary, collect, key_chain=None):
    result = OrderedDict()
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
        __walk_key_chain(d, ['a', 'b', 'c'])  # returns 15
    :param dictionary: a nested dictionary containing other dictionaries
    :param key_chain: a list of keys to traverse down the nesting structure
    :return: the value in the nested structure after traversing down the `key_chain`
    """
    return reduce(lambda d, k: d[k], key_chain, dictionary)


def proportion_different(weights1, weights2):
    """
    Returns the number of weights that changed across all layers
    divided by the total number of weights across all layers.
    """
    assert weights1.keys() == weights2.keys()
    num_weights = 0
    num_weights_changed = 0

    def count_weights(key_chain, w1):
        w2 = walk_key_chain(weights2, key_chain)
        assert w2.size == w1.size
        nonlocal num_weights, num_weights_changed
        num_weights += w1.size
        num_weights_changed += (w1 != w2).sum()

    walk(weights1, collect=count_weights)
    return num_weights_changed / num_weights


def validate_weights(weight_names):
    exists = [os.path.isfile(os.path.join("weights", weight + ".h5")) for weight in weight_names]
    assert all(exists), "weights do not exist: %s" % ", ".join(
        weight for (weight, exist) in zip(weight_names, exists) if not exist)
