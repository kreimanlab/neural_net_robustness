from convnetskeras.convnets import preprocess_image_batch, AlexNet
from keras.applications import ResNet50, VGG16, VGG19, InceptionV3
from keras.applications.imagenet_utils import preprocess_input
from keras.optimizers import SGD
from keras.preprocessing import image
import numpy as np


def alexnet(weights_path=None, metrics=[]):
    model = AlexNet(weights_path)
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='mse', metrics=metrics)
    model.preprocess_images = lambda image_paths: \
        preprocess_image_batch(image_paths, img_size=(256, 256), crop_size=(227, 227), color_mode="rgb")
    return model


def resnet50(weights=None):
    return _keras_model(ResNet50(weights=weights))


def vgg16(weights=None):
    return _keras_model(VGG16(weights=weights))


def vgg19(weights=None):
    return _keras_model(VGG19(weights=weights))


def inceptionv3(weights=None):
    return _keras_model(InceptionV3(weights=weights))


def _keras_model(model):
    model.preprocess_images = _preprocess_keras_images
    return model


def _preprocess_keras_images(image_paths):
    preprocessed_images = []
    for image_path in image_paths:
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preprocessed_images.append(x)
    return preprocessed_images
