import glob
import os
import re


def get_predictions_filepath(dataset, weights, variations=False, predictions_directory="predictions"):
    filepath = os.path.join(predictions_directory, dataset, weights + ".p")
    return get_files(filepath, variations)


def get_files(filepath, variations=False, directory=''):
    """
    :param variations: either False to not search for variations, True to search for variations
    or an integer (array) to directly specify the variations.
    If variations are not being searched, the file directory will also be created.
    """
    filepath_without_extension, extension = os.path.splitext(filepath)
    file_directory = os.path.dirname(os.path.join(directory, filepath))
    if variations is False:  # no variations
        os.makedirs(file_directory, exist_ok=True)
        return filepath
    if isinstance(variations, int) and variations is not True:  # exact variations specified
        os.makedirs(file_directory, exist_ok=True)
        return "%s-num%d%s" % (filepath_without_extension, variations, extension)
    try:  # iterable varations
        if all(isinstance(num, int) for num in variations):
            os.makedirs(file_directory, exist_ok=True)
            return ["%s-%d%s" % (filepath_without_extension, num, extension) for num in variations], variations
    except TypeError:
        pass
    if os.path.isfile(os.path.join(directory, filepath)):  # ignore variations if file itself exists
        os.makedirs(file_directory, exist_ok=True)
        num = re.search(r'-num([0-9]+)' + extension + '$', filepath)
        if num is not None:
            num = int(num.group(1))
        return [filepath], [num]
    # search variations on filesystem
    pattern = r'^%s-num([0-9]+)%s$' % (re.escape(os.path.basename(filepath_without_extension)), extension)
    nums = [int(re.search(pattern, os.path.basename(file)).group(1))
            for file in glob.glob(os.path.join(directory, "%s-num*%s" % (filepath_without_extension, extension)))
            if re.search(pattern, os.path.basename(file))]
    return ["%s-num%d%s" % (filepath_without_extension, num, extension)
            for num in nums], nums
