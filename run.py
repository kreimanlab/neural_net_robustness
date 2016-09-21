import argparse
import functools
import os
import pickle

import itertools
import numpy as np

from datasets import get_data
from net import alexnet, preprocess_images_alexnet
from results import get_results_filepath


def __adapt_to_softmax(x, length):
    try:
        len(x)
    except TypeError:
        x = [x]
    softmaxs = np.zeros([len(x), length])
    for row, col in enumerate(x):
        softmaxs[row, col] = 1
    return softmaxs


def __generator(image_paths, truth_labels, load_image, output_length):
    assert len(truth_labels) > 0
    assert len(truth_labels) == len(image_paths)
    for image_path, truth_label in itertools.cycle(zip(image_paths, truth_labels)):
        image = load_image(image_path)
        truth = __adapt_to_softmax(truth_label, output_length)
        yield image, truth


def retrain(model, datasets_directory, datasets, data_generator, num_epochs=10):
    for retrain_dataset in datasets:
        print("Collect %s" % retrain_dataset)
        image_paths, truth_labels = get_data(os.path.join(datasets_directory, retrain_dataset))
        generator = data_generator(image_paths, truth_labels)
        print("Retrain on %s" % retrain_dataset)
        # TODO: use best weights based on validation error
        model.fit_generator(generator, samples_per_epoch=len(truth_labels), nb_epoch=num_epochs)
        weights_file = "weights/retrain/%s/%s/%depochs.h5" % (args.model, retrain_dataset, args.num_epochs)
        print("Save weights to %s" % weights_file)
        os.makedirs(os.path.dirname(weights_file), exist_ok=True)
        model.save_weights(weights_file)


def evaluate(model, weights_names, datasets_directory, datasets, data_generator):
    # TODO: we need to evaluate this with a SVM to get rid of errors due to re-ordering of labels
    for dataset_name in datasets:
        print("Collect %s" % dataset_name)
        image_paths, truth_labels = get_data(os.path.join(datasets_directory, dataset_name))
        for weights_name in weights_names:
            print("Evaluating with %s" % weights_name)
            model.load_weights("weights/%s.h5" % weights_name)
            generator = data_generator(image_paths, truth_labels)
            metric = model.evaluate_generator(generator, val_samples=len(truth_labels), max_q_size=1000)
            results_filepath = get_results_filepath(dataset_name, weights_name)
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
    parser.add_argument('--datasets_directory', type=str, default='datasets',
                        help='The directory all datasets are stored in')
    parser.add_argument('--datasets', type=str, nargs='+', default=['ILSVRC2012/test'],
                        help='The datasets to either re-train or evaluate the model with')
    parser.add_argument('--weights', type=str, nargs='+', default=None,
                        help='The set of weights to evaluate the model with - re-trains the model if this is not set')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='how many epochs to search for optimal weights during training')
    args = parser.parse_args()
    print('Running with args', args)
    weights = args.weights
    assert weights is None or all(os.path.isfile(os.path.join("weights", weight + ".h5")) for weight in weights)
    datasets = args.datasets
    assert datasets is not None and \
           all(os.path.isdir(os.path.join(args.datasets_directory, dataset))
               and os.path.isfile(os.path.join(args.datasets_directory, dataset, "ground_truths.p"))
               for dataset in datasets)
    # model
    model = models[args.model]()
    output_shape = model.get_output_shape_at(-1)
    generator = functools.partial(__generator,
                                  load_image=lambda path: image_preprocessors[args.model]([path]),
                                  output_length=output_shape[1])
    if weights is None:
        print("Retraining")
        model.load_weights("weights/%s.h5" % args.model)
        retrain(model, args.datasets_directory, datasets, data_generator=generator, num_epochs=args.num_epochs)
    else:
        print("Evaluating")
        evaluate(model, weights, args.datasets_directory, datasets, data_generator=generator)
