import glob
import os
import re


def get_results_filepath(dataset, weights, search_variations=False):
    filepath_without_extension = "results/%s/%s" % (dataset, weights)
    filepath = "%s.p" % filepath_without_extension
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if search_variations:
        if not os.path.isfile(filepath):
            pattern = r'^%s-num[0-9]+.p$' % re.escape(os.path.basename(filepath_without_extension))
            return [file for file in glob.glob("%s-num*.p" % filepath_without_extension)
                    if re.search(pattern, os.path.basename(file))]
        return [filepath]
    return filepath

