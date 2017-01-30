import os
from collections import OrderedDict, Counter
from functools import reduce

import h5py
import numpy as np

from predictions import get_files


def get_weights_filepath(weights_name, variations=False, weights_directory='weights'):
    return get_files(weights_name + ".h5", variations=variations, directory=weights_directory)


def sort_weights_by(weights, layer_sorting):
    assert Counter(layer_sorting) == Counter(weights.keys())
    sorted_weights = OrderedDict()
    for layer in layer_sorting:
        sorted_weights[layer] = weights[layer]
    return sorted_weights


def load_weights(*weights_names, keep_names=False, weights_directory='weights'):
    weights = list()
    for weights_name in weights_names:
        filepath = "%s/%s%s" % (weights_directory, weights_name, '.h5' if not weights_name.endswith('.h5') else '')
        with h5py.File(filepath, 'r') as file:
            w = walk(file, lambda _, x: np.array(x))
            if 'layer_names' in file.attrs:
                # keras reordering
                sorted_layers = [l.decode('utf8') for l in file.attrs['layer_names']]
                w = sort_weights_by(w, sorted_layers)
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


def validate_weights(weight_names, weights_directory="weights"):
    weight_paths = [os.path.join(weights_directory, weight + ".h5") for weight in weight_names]
    exists = [os.path.isfile(weight) for weight in weight_paths]
    assert all(exists), "weights do not exist: %s" % ", ".join(
        weight_path for (weight_path, exist) in zip(weight_paths, exists) if not exist)


def _is_sub_layer(layer, parent_layer):
    return layer.startswith(parent_layer) and layer != parent_layer


def has_sub_layers(weights, layer):
    return any(_is_sub_layer(weights_name, layer) for weights_name in weights)


def merge_sub_layers(weights, layer):
    W, b = [], []
    for weights_name, weight_values in weights.items():
        if _is_sub_layer(weights_name, layer):
            W.append(weights[weights_name][weights_name + '_W'])
            b.append(weights[weights_name][weights_name + '_b'])
    return {layer + '_W': np.array(W), layer + '_b': np.array(b)}
