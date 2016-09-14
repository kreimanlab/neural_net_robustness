import argparse
import os
import numpy as np

from datasets import get_data
from net import alexnet, preprocess_images_alexnet


def adapt_to_softmax(x, length):
    softmaxs = np.zeros([len(x), length])
    for row, col in enumerate(x):
        softmaxs[row, col] = 1
    return softmaxs


def prepare_data(data_directory, collect_images, output_shape):
    image_paths, truths = get_data(data_directory)
    assert len(image_paths) > 0
    assert len(truths) > 0
    truths = truths[:len(image_paths)]  # for local testing where we do not have all images TODO: remove
    images = collect_images(image_paths)
    assert len(truths) == len(image_paths)
    truths = adapt_to_softmax(truths, output_shape[1])
    return images, truths


if __name__ == '__main__':
    # options
    models = {'alexnet': alexnet}
    # params - command line
    parser = argparse.ArgumentParser(description='Neural Net Robustness')
    parser.add_argument('--model', type=str, default=next(models.__iter__()),
                        help='The model to run', choices=models.keys())
    parser.add_argument('--retrain_dataset', type=str, default='VOC2012',
                        help='With what dataset to re-train',
                        choices=[d for d in os.listdir("datasets") if os.path.isdir(os.path.join("datasets", d))])
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='how many epochs to search for optimal weights during training')
    args = parser.parse_args()
    print('Running with args', args)
    # model
    model_provider = models[args.model]
    model = model_provider(weights_path="weights/%s.h5" % args.model, metrics=['accuracy'])
    output_shape = model.get_output_shape_at(-1)
    initial_weights = model.get_weights()
    # re-train
    retrain_dataset = args.retrain_dataset
    print("Collect %s" % retrain_dataset)
    retrain_images, retrain_truths = prepare_data("datasets/%s" % retrain_dataset,
                                                  collect_images=preprocess_images_alexnet, output_shape=output_shape)
    print("Retrain on %s" % retrain_dataset)
    model.fit(retrain_images, retrain_truths, nb_epoch=args.num_epochs)
    print("Save weights")
    model.save_weights("weights/%s_retrained_on_%s_for_%d_epochs.h5" % (args.model, retrain_dataset, args.num_epochs))
