import argparse
import itertools
import os
import pickle

import functools
import numpy as np

from datasets import get_data, validate_datasets
from net import alexnet, preprocess_images_alexnet
from predictions import get_predictions_filepath
from weights import validate_weights


def __adapt_to_softmax(x, length):
    try:
        len(x)
    except TypeError:
        x = [x]
    softmaxs = np.zeros([len(x), length])
    for row, col in enumerate(x):
        softmaxs[row, col] = 1
    return softmaxs


def __image_truth_generator(truths_mapping, datasets_directory, load_image, output_length, path_output=None):
    assert len(truths_mapping) > 0
    for image_path, truth_label in itertools.cycle(truths_mapping.items()):
        image = load_image(os.path.join(datasets_directory, image_path))
        truth = __adapt_to_softmax(truth_label, output_length)
        yield image, truth
        if path_output is not None:
            path_output.append(image_path)


def __image_generator(image_truth_generator):
    for image, _ in image_truth_generator:
        yield image


def retrain(model, datasets, image_truth_generator,
            num_epochs=10, image_batch_size=1000, datasets_directory="datasets", weights_directory="weights"):
    for retrain_dataset in datasets:
        print("Collect %s" % retrain_dataset)
        truths_mapping = get_data(retrain_dataset, datasets_directory)
        generator = image_truth_generator(truths_mapping, datasets_directory=datasets_directory)
        print("Retrain on %s" % retrain_dataset)
        # TODO: use best weights based on validation error
        model.fit_generator(generator, samples_per_epoch=len(truths_mapping), nb_epoch=num_epochs,
                            max_q_size=image_batch_size)
        weights_file = "%s/retrain/%s/%s/%depochs.h5" % (
            weights_directory, args.model, retrain_dataset, args.num_epochs)
        print("Save weights to %s" % weights_file)
        os.makedirs(os.path.dirname(weights_file), exist_ok=True)
        model.save_weights(weights_file)


def predict(model, weights_names, datasets, image_truth_generator, image_batch_size=1000,
            weights_directory="weights", datasets_directory="datasets", predictions_directory="predictions"):
    for dataset_name in datasets:
        print("Collect %s" % dataset_name)
        truths_mapping = get_data(dataset_name, datasets_directory)
        for weights_name in weights_names:
            print("Predicting with %s" % weights_name)
            model.load_weights("%s/%s.h5" % (weights_directory, weights_name))
            image_paths = []
            generator = image_truth_generator(truths_mapping, datasets_directory=datasets_directory,
                                              path_output=image_paths)
            generator = __image_generator(generator)
            predictions = model.predict_generator(generator, val_samples=len(truths_mapping),
                                                  max_q_size=image_batch_size)
            mapped_predictions = dict((image, prediction) for image, prediction
                                      in zip(image_paths[:len(predictions)], predictions))
            results_filepath = get_predictions_filepath(dataset_name, weights_name,
                                                        predictions_directory=predictions_directory)
            print("Writing predictions to %s" % results_filepath)
            with open(results_filepath, 'wb') as results_file:
                pickle.dump({'predictions': mapped_predictions, 'dataset': dataset_name, 'weights': weights_name},
                            results_file)


if __name__ == '__main__':
    # options
    models = {'alexnet': alexnet}
    image_preprocessors = {'alexnet': preprocess_images_alexnet}
    # params - command line
    parser = argparse.ArgumentParser(description='Neural Net Robustness')
    parser.add_argument('--model', type=str, default=next(models.__iter__()),
                        help='The model to run', choices=models.keys())
    parser.add_argument('--weights_directory', type=str, default='weights',
                        help='The directory in which the weights are stored in')
    parser.add_argument('--datasets_directory', type=str, default='datasets',
                        help='The directory in which the datasets are stored in')
    parser.add_argument('--predictions_directory', type=str, default='predictions',
                        help='The directory all predictions should be stored in')
    parser.add_argument('--datasets', type=str, nargs='+', default=['ILSVRC2012/val'],
                        help='The datasets to either re-train the model with or to predict')
    parser.add_argument('--weights', type=str, nargs='+', default=None,
                        help='The set of weights to use for prediction - re-trains the model if this is not set')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='how many epochs to search for optimal weights during training')
    parser.add_argument('--image_batch_size', type=int, default=1000,
                        help='how many images to load into memory at once')
    args = parser.parse_args()
    print('Running with args', args)
    weights = args.weights
    if weights is not None:
        validate_weights(weights, args.weights_directory)
    datasets = args.datasets
    validate_datasets(datasets, args.datasets_directory)
    # model
    model = models[args.model]()
    output_shape = model.get_output_shape_at(-1)
    generator = functools.partial(__image_truth_generator,
                                  load_image=lambda path: image_preprocessors[args.model]([path]),
                                  output_length=output_shape[1])
    if weights is None:
        print("Retraining")
        model.load_weights("weights/%s.h5" % args.model)
        retrain(model, datasets, num_epochs=args.num_epochs,
                image_truth_generator=generator, image_batch_size=args.image_batch_size,
                datasets_directory=args.datasets_directory, weights_directory=args.weights_directory)
    else:
        print("Predicting")
        predict(model, weights, datasets,
                image_truth_generator=generator, image_batch_size=args.image_batch_size,
                weights_directory=args.weights_directory,
                datasets_directory=args.datasets_directory,
                predictions_directory=args.predictions_directory)
