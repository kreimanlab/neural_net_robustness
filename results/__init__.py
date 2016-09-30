from predictions import get_outputs


def get_results_filepath(dataset, weights, variations=False):
    return get_outputs("results", dataset, weights, variations)
