import glob
from numbers import Number


def get_data(directory, extensions=['jpg', 'jpeg', 'JPG', 'JPEG', 'png', 'PNG']):
    image_paths = [f for extension in extensions for f in glob.glob("%s/*.%s" % (directory, extension))]
    with open("%s/ground_truth.txt" % directory, 'r') as truths_file:
        truths = truths_file.read().splitlines()
    if not all(isinstance(name, Number) for name in truths):
        unique = sorted(set(truths))
        indexed_truths = list()
        for name in truths:
            indexed_truths.append(unique.index(name))
        truths = indexed_truths
    return image_paths, truths
