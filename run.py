import argparse
import functools
import os
import pickle

import numpy as np

from datasets import get_data
from net import alexnet, preprocess_images_alexnet


def __adapt_to_softmax(x, length):
    softmaxs = np.zeros([len(x), length])
    for row, col in enumerate(x):
        softmaxs[row, col] = 1
    return softmaxs


def __prepare_data(data_directory, collect_images, output_shape):
    image_paths, truths = get_data(data_directory)
    assert len(image_paths) > 0
    assert len(truths) > 0
    images = collect_images(image_paths)
    assert len(truths) == len(image_paths)
    truths = __adapt_to_softmax(truths, output_shape[1])
    return images, truths


def retrain(model, datasets, prepare_data):
    for retrain_dataset in datasets:
        print("Collect %s" % retrain_dataset)
        images, truths = prepare_data(data_directory="datasets/%s" % retrain_dataset)
        print("Retrain on %s" % retrain_dataset)
        model.fit(images, truths, nb_epoch=args.num_epochs)
        weights_file = "weights/%s_retrained_on_%s_for_%d_epochs.h5" % (args.model, retrain_dataset, args.num_epochs)
        print("Save weights to %s" % weights_file)
        model.save_weights(weights_file)


def __fix_evaluation_metric(metric):
    """
    Due to a bug in keras, the metric always has a leading 0.0 which this method removes
    :param metric: the metric to fix containing two values where the first is a 0
    :return: the last value of the metric
    """
    assert len(metric) == 2
    return metric[-1]


def evaluate(model, weights_names, datasets, prepare_data):
    # TODO: we need to evaluate this with a SVM to get rid of errors due to re-ordering of labels
    for dataset_name in datasets:
        print("Collect %s" % dataset_name)
        images, truths = prepare_data(data_directory="datasets/%s" % dataset_name)
        for weights_name in weights_names:
            print("Evaluating with %s" % weights_name)
            model.load_weights("weights/%s.h5" % weights_name)
            metric = model.evaluate(images, truths)
            metric = __fix_evaluation_metric(metric)
            results_filepath = "results/%s-%s.p" % (dataset_name, weights_name)
            print("Writing results to %s" % results_filepath)
            with open(results_filepath, 'wb') as results_file:
                pickle.dump({'metric': metric, 'dataset': dataset_name, 'weights': weights_name}, results_file)


if __name__ == '__main__':
    # options
    models = {'alexnet': alexnet}
    image_preprocessors = {'alexnet': preprocess_images_alexnet}
    # params - command line
    parser = argparse.ArgumentParser(description='Neural Net Robustness')
    parser.add_argument('--model', type=str, default=next(models.__iter__()),
                        help='The model to run', choices=models.keys())
    parser.add_argument('--datasets', type=str, nargs='+', default=['VOC2012'],
                        help='The datasets to either re-train or evaluate the model with',
                        choices=[d for d in os.listdir("datasets") if os.path.isdir(os.path.join("datasets", d))])
    parser.add_argument('--weights', type=str, nargs='+', default=None,
                        help='The set of weights to evaluate the model with - re-trains the model if this is not set')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='how many epochs to search for optimal weights during training')
    args = parser.parse_args()
    print('Running with args', args)
    weights = args.weights
    datasets = args.datasets
    # model
    model = models[args.model](metrics=['accuracy'])
    output_shape = model.get_output_shape_at(-1)
    prepare_data = functools.partial(__prepare_data,
                                     collect_images=image_preprocessors[args.model],
                                     output_shape=output_shape)
    if weights is None:
        print("Retraining")
        model.load_weights("weights/%s.h5" % args.model)
        retrain(model, datasets, prepare_data)
    else:
        print("Evaluating")
        evaluate(model, weights, datasets, prepare_data)
