import os
import pickle
from collections import OrderedDict
from numbers import Number


def validate_datasets(datasets, datasets_directory="datasets"):
    assert datasets is not None, "datasets not set"
    dirs_exist = [os.path.isdir(os.path.join(datasets_directory, dataset))
                  for dataset in datasets]
    truths_exist = [os.path.isfile(os.path.join(datasets_directory, dataset, "ground_truths.p"))
                    for dataset in datasets]
    assert all(dirs_exist) and all(truths_exist), "dataset directory or ground_truths.p does not exist: %s" % ", ".join(
        dataset for (dataset, dir_exist, truth_exist) in zip(datasets, dirs_exist, truths_exist)
        if not dir_exist or not truth_exist)


def get_data(dataset, datasets_directory="datasets"):
    truths_mapping = pickle.load(open("%s/%s/ground_truths.p" % (datasets_directory, dataset), 'rb'))
    truths = truths_mapping.values()
    if not all(isinstance(truth, Number) for truth in truths):
        try:
            truths_conversion = dict((truth, int(truth)) for truth in truths)
        except ValueError:
            truths_conversion = __index_names(truths)
    else:
        truths_conversion = dict((truth, truth) for truth in truths)
    truths_mapping = OrderedDict(("%s/images/%s" % (dataset, image_file), truths_conversion[truth])
                                 for image_file, truth in truths_mapping.items())
    return truths_mapping


def __index_names(truth_names):
    unique = sorted(set(truth_names))
    indexed_truths = {}
    for name in truth_names:
        indexed_truths[name] = unique.index(name)
    return indexed_truths
