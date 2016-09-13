import numpy as np

from datasets import get_data
from net import alexnet, preprocess_images_alexnet


def train(model, images, truth):
    model.fit(images, truth)


def test(model, images, truth):
    return model.evaluate(images, truth)


def adapt_to_softmax(x, length):
    softmaxs = np.zeros([len(x), length])
    for row, col in enumerate(x):
        softmaxs[row, col] = 1
    return softmaxs


def prepare_data(data_directory, collect_images, output_shape):
    image_paths, truths = get_data(data_directory)
    assert len(image_paths) > 0
    assert len(truths) > 0
    images = collect_images(image_paths)
    assert len(truths) == len(image_paths)
    truths = adapt_to_softmax(truths, output_shape[1])
    return images, truths


if __name__ == '__main__':
    model = alexnet(weights_path="weights/alexnet_weights.h5", metrics=['accuracy'])
    output_shape = model.get_output_shape_at(-1)
    initial_weights = model.get_weights()
    # re-train on VOC2012
    print("Collect VOC2012")
    voc2012images, voc2012truths = prepare_data("datasets/VOC2012",
                                                collect_images=preprocess_images_alexnet, output_shape=output_shape)
    print("Retrain on VOC2012")
    train(model, voc2012images, voc2012truths)
    print("Save weights")
    model.save_weights("weights/retrain_voc2012.h5")
