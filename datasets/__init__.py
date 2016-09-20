import os
import pickle
from numbers import Number


def get_data(directory):
    truths_mapping = pickle.load(open("%s/ground_truths.p" % directory, 'rb'))
    image_paths = [os.path.join(directory, "images", image_file) for image_file in truths_mapping.keys()]
    truths = list(truths_mapping.values())

    if not all(isinstance(name, Number) for name in truths):
        unique = sorted(set(truths))
        indexed_truths = list()
        for name in truths:
            indexed_truths.append(unique.index(name))
        truths = indexed_truths
    return image_paths, truths
