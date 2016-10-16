import argparse
import copy
import functools
import math
import random
from collections import OrderedDict

import h5py
import numpy as np

from weights import load_weights, validate_weights, merge_sub_layers
from weights.analyze import proportion_different


def __dump_weights_to_hdf5(weights, filepath, base_weights_name=None):
    """
    Save the given weights in HDF5 format so that they can be loaded by a model.
    Adopted from keras.engine.topology.Container#save_weights_to_hdf5_group
    """
    # We cannot use the Container save method directly since it uses the weights set in the backend.
    # We also cannot set the weights of the container since it expects a properly ordered numpy array.
    with h5py.File(filepath, 'w') as file:
        if base_weights_name is not None:
            file.attrs['base_weights'] = base_weights_name
        file.attrs['layer_names'] = [layer.encode('utf8') for layer in weights]
        for layer in weights:
            group = file.create_group(layer)
            group.attrs['weight_names'] = [name.encode('utf8') for name in weights[layer]]
            for name, val in weights[layer].items():
                param_dset = group.create_dataset(name.encode('utf8'), val.shape, dtype=val.dtype)
                if not val.shape:  # scalar
                    param_dset[()] = val
                else:
                    param_dset[:] = val


def __draw(x, proportion):
    num_elements = math.floor(proportion * x.size)
    x_prime = x.reshape(x.size)
    indices = random.sample(range(x.size), num_elements)
    x_prime[indices] = random.sample(list(x_prime), num_elements)
    return x_prime.reshape(x.shape)


def __mutate_gaussian(x, proportion):
    # Note: proportion is used in a different context than in __draw,
    # but is still the primary parameter of weight perturbations
    # It would help clarity of arg names to use different names for the different uses,
    # but this was omitted for sake of succinctness.
    return x + np.random.normal(loc=0.0, scale=proportion * np.std(x), size=x.shape)


def __divide_sub_weights(target_weights, source_weights, layer):
    for weights_name in target_weights:
        if weights_name.startswith(layer) and weights_name != layer:
            num = int(weights_name.split('_')[-1]) - 1
            for weight_type in ['W', 'b']:
                source_values = source_weights['_'.join(weights_name.split('_')[:-1]) + '_' + weight_type][num]
                assert target_weights[weights_name][weights_name + '_' + weight_type].shape == source_values.shape
                target_weights[weights_name][weights_name + '_' + weight_type] = source_values


def __perturb_all(weights, layer, perturb_func):
    """
    Perturbs all weights in the `layer` using the `perturb_func`.
    If the weights in this layer are divided, e.g. for layer 'conv_2' into 'conv_2_1' and 'conv_2_2',
    merges all "sub-weights", perturbs them and divides them again according to the previous structure.
    """
    layer_weights = weights[layer]
    if not layer_weights:  # sub-weights
        layer_weights = merge_sub_layers(weights, layer)

    perturbed_weights = OrderedDict()
    for weight_name, weight_values in layer_weights.items():
        # we want to change W and b separately because their distributions can be vastly different
        perturbed_weights[weight_name] = perturb_func(weight_values)

    if weights[layer]:  # no sub-weights
        for weight_name in weights[layer]:  # assign directly to avoid any dict-reordering
            weights[layer][weight_name] = perturbed_weights[weight_name]
    else:  # has sub-weights
        __divide_sub_weights(weights, perturbed_weights, layer)


def main():
    # options
    perturbations = {'draw': __draw, 'mutateGaussian': __mutate_gaussian}
    # params - command line
    parser = argparse.ArgumentParser(description='Neural Net Robustness - Weight Perturbation')
    parser.add_argument('--weights', type=str, nargs='+', default=['alexnet'],
                        help='The weights to perturb')
    parser.add_argument('--weights_directory', type=str, default='weights',
                        help='The directory in which the weights are stored in')
    parser.add_argument('--layer', type=str, nargs='+', default=['conv_1'],
                        help='In what layer(s) to perturb the weights')
    parser.add_argument('--perturbation', type=str, nargs='+', default=[next(perturbations.__iter__())],
                        help='How to perturb the weights', choices=perturbations.keys())
    parser.add_argument('--proportion', type=float, nargs='+', default=np.arange(0.1, 0.6, 0.1),
                        help='In draw: what proportion(s) of the weights to perturb; '
                             'In mutateGaussian: the number of proportion of the standard distribution(s) '
                             'of the current weights to use as the standard distrbution of the weight perturbations(s)')
    parser.add_argument('--num_perturbations', type=int, default=1,
                        help='How often to perturb the weights in different variations')
    args = parser.parse_args()
    print('Running with args', args)
    validate_weights(args.weights, args.weights_directory)
    weights = load_weights(*args.weights, keep_names=True, weights_directory=args.weights_directory)
    random.seed(0)
    # perturb
    for weight_name, weight_values in weights.items():
        for perturbation_name in args.perturbation:
            perturb = perturbations[perturbation_name]
            for layer in args.layer:
                for proportion in args.proportion:
                    for nth_perturbation in range(1, args.num_perturbations + 1):
                        print('Perturbing %s.%s with %.2f %s (%d/%d)' %
                              (weight_name, layer, proportion, perturbation_name, nth_perturbation,
                               args.num_perturbations))
                        perturbed_weights = copy.deepcopy(weight_values)
                        __perturb_all(perturbed_weights, layer, functools.partial(perturb, proportion=proportion))
                        assert proportion_different(weight_values, perturbed_weights, mean_across_layers=True) > 0, \
                            "No weights changed"
                        save_filepath = '%s/perturbations/%s-%s-%s%.2f-num%d.h5' % (
                            args.weights_directory, weight_name, layer, perturbation_name, proportion, nth_perturbation)
                        __dump_weights_to_hdf5(perturbed_weights, save_filepath, base_weights_name=weight_name)
                        print('Saved to %s' % save_filepath)


if __name__ == '__main__':
    main()
