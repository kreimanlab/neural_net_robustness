import glob
import os
import re


def get_predictions_filepath(dataset, weights, variations=False):
    return get_outputs("predictions", dataset, weights, variations)


def get_outputs(directory, dataset, weights, variations=False):
    """
    :param variations: either False to not search for variations, True to search for variations,
    a single integer to use this variation or an integer array to directly specify the variations
    """
    filepath_without_extension = "%s/%s/%s" % (directory, dataset, weights)
    filepath = "%s.p" % filepath_without_extension
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if variations is False:
        return filepath
    if isinstance(variations, int) and variations is not True:
        return "%s-num%d.p" % (filepath_without_extension, variations)
    try:
        if all(isinstance(num, int) for num in variations):
            return ["%s-%d.p" % (filepath_without_extension, num) for num in variations], variations
    except TypeError:
        pass
    if os.path.isfile(filepath):  # ignore variations if file itself exists
        return [filepath], [None]
    pattern = r'^%s-num([0-9]+).p$' % re.escape(os.path.basename(filepath_without_extension))
    nums = [int(re.search(pattern, os.path.basename(file)).group(1))
            for file in glob.glob("%s-num*.p" % filepath_without_extension)
            if re.search(pattern, os.path.basename(file))]
    return ["%s-num%d.p" % (filepath_without_extension, num)
            for num in nums], nums
