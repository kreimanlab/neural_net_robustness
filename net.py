from convnetskeras.convnets import preprocess_image_batch, AlexNet
from keras.optimizers import SGD


def alexnet(weights_path=None, metrics=[]):
    model = AlexNet(weights_path)

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='mse', metrics=metrics)
    return model


def preprocess_images_alexnet(image_paths):
    return preprocess_image_batch(image_paths,
                                  img_size=(256, 256), crop_size=(227, 227), color_mode="rgb")
