import os

from predictions import get_files


def get_results_filepath(dataset, weights, variations=False, results_directory="results"):
    filepath = os.path.join(results_directory, dataset, weights + ".p")
    return get_files(filepath, variations)
