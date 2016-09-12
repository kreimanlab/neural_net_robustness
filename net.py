from convnetskeras.customlayers import crosschannelnormalization, \
    splittensor
from keras.layers import Flatten, Dense, Dropout, Activation, \
    Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.models import Model
from keras.optimizers import SGD


def alexnet(weights_path=None):
    inputs = Input(shape=(3, 227, 227))

    conv_1 = Convolution2D(96, 11, 11, subsample=(4, 4), activation='relu',
                           name='conv_1')(inputs)
    conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1)
    conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
    conv_2 = ZeroPadding2D((2, 2))(conv_2)
    conv_2 = merge([
                       Convolution2D(128, 5, 5, activation="relu", name='conv_2_' + str(i + 1))(
                           splittensor(ratio_split=2, id_split=i)(conv_2)
                       ) for i in range(2)], mode='concat', concat_axis=1, name="conv_2")

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_3 = Convolution2D(384, 3, 3, activation='relu', name='conv_3')(conv_3)

    conv_4 = ZeroPadding2D((1, 1))(conv_3)
    conv_4 = merge([
                       Convolution2D(192, 3, 3, activation="relu", name='conv_4_' + str(i + 1))(
                           splittensor(ratio_split=2, id_split=i)(conv_4)
                       ) for i in range(2)], mode='concat', concat_axis=1, name="conv_4")

    conv_5 = ZeroPadding2D((1, 1))(conv_4)
    conv_5 = merge([
                       Convolution2D(128, 3, 3, activation="relu", name='conv_5_' + str(i + 1))(
                           splittensor(ratio_split=2, id_split=i)(conv_5)
                       ) for i in range(2)], mode='concat', concat_axis=1, name="conv_5")

    dense_6 = MaxPooling2D((3, 3), strides=(2, 2), name="convpool_5")(conv_5)

    dense_6 = Flatten(name="flatten")(dense_6)
    dense_6 = Dense(4096, activation='relu', name='fc6')(dense_6)
    dense_7 = Dropout(0.5)(dense_6)
    dense_7 = Dense(4096, activation='relu', name='fc7')(dense_7)
    dense_8 = Dropout(0.5)(dense_7)
    dense_8 = Dense(1000, name='fc8')(dense_8)
    prediction = Activation("softmax", name="softmax")(dense_8)

    model = Model(input=inputs, output=prediction)

    if weights_path:
        model.load_weights(weights_path)

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='mse')
    return model
