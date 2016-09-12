import numpy as np
import scipy.io
from convnetskeras.convnets import preprocess_image_batch

from net import alexnet

if __name__ == '__main__':
    words = scipy.io.loadmat("datasets/ILSVRC2012/synset_words.mat")
    words = words['synset']
    words = [t[1] for t in words[:, 0]]
    model = alexnet(weights_path="alexnet_weights.h5")
    img = preprocess_image_batch(['datasets/dog.jpg'],
                                 img_size=(256, 256), crop_size=(227, 227), color_mode="rgb")
    out = model.predict(img)
    print(words[np.argmax(out)])
