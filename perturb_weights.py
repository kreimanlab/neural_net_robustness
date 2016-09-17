import argparse
import math
import random

import h5py

from weights import load_weights


def __dump_weights_to_hdf5(weights, filepath):
    """
    Save the given weights in HDF5 format so that they can be loaded by a model.
    Adopted from keras.engine.topology.Container#save_weights_to_hdf5_group
    """
    # We cannot use the Container save method directly since it uses the weights set in the backend.
    # We also cannot set the weights of the container since it expects a properly ordered numpy array.
    with h5py.File(filepath, 'w') as file:
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


if __name__ == '__main__':
    # options
    perturbations = {'draw': __draw}
    # params - command line
    parser = argparse.ArgumentParser(description='Neural Net Robustness - Weight Perturbation')
    parser.add_argument('--weights', type=str, nargs='+', default=['alexnet'],
                        help='The weights to perturb')
    parser.add_argument('--layer', type=str, default=['conv_1'],
                        help='In what layer(s) to perturb the weights')
    parser.add_argument('--perturbation', type=str, nargs='+', default=[next(perturbations.__iter__())],
                        help='How to perturb the weights')
    parser.add_argument('--ratio', type=float, nargs='+', default=[.1],
                        help='What ratio(s) of the weights to perturb')
    parser.add_argument('--num_perturbations', type=int, default=1,
                        help='How often to perturb the weights in different variations')
    args = parser.parse_args()
    print('Running with args', args)
    weights = load_weights(*args.weights, keep_names=True)
    random.seed(0)
    # perturb
    for weight_name, weight_values in weights.items():
        for perturbation_name in args.perturbation:
            perturb = perturbations[perturbation_name]
            for layer in args.layer:
                for ratio in args.ratio:
                    for nth_perturbation in range(1, args.num_perturbations + 1):
                        print('Perturbing %s.%s with %.2f %s (%d/%d)' %
                              (weight_name, layer, ratio, perturbation_name, nth_perturbation, args.num_perturbations))
                        perturbed_weights = weight_values
                        for layer_weight_name, layer_weight_values in weight_values[layer].items():
                            # we want to change W and b separately because their distributions can be vastly different
                            perturbed_weights[layer][layer_weight_name] = perturb(
                                perturbed_weights[layer][layer_weight_name], ratio)
                        save_filepath = 'weights/perturbations/%s-%s-%s%.2f-num%d.h5' % \
                                        (weight_name, layer, perturbation_name, ratio, nth_perturbation)
                        __dump_weights_to_hdf5(perturbed_weights, save_filepath)
                        print('Saved to %s' % save_filepath)
