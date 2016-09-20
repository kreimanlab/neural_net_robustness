import argparse
import glob
import os
import pickle
import shutil
import tarfile
import xml.etree.ElementTree as xml
from urllib.request import urlretrieve


def __retrieve_tarred_content(remote_url, local_path):
    assert remote_url.endswith(".tar.gz") or remote_url.endswith(".tar")
    if os.path.exists(local_path):
        print("Local path %s already exists" % local_path)
        return
    tar_path = "%s.%s" % (local_path, "tar.gz" if remote_url.endswith(".tar.gz") else "tar")
    __download_if_needed(remote_url, tar_path)
    __extract_tar(tar_path, local_path)


def __download_if_needed(url, local_path):
    download_needed = not os.path.isfile(local_path)
    if download_needed:
        print("Downloading %s..." % local_path)
        filepart_path = "%s.filepart" % local_path
        urlretrieve(url, filepart_path)
        shutil.move(filepart_path, local_path)
    else:
        print("Not downloading %s (exists already)" % local_path)
    return download_needed


def __extract_tar(filepath, target_directory="."):
    print("Untarring %s to %s..." % (filepath, target_directory))
    tar = tarfile.open(filepath, "r:gz" if filepath.endswith(".tar.gz") else "r:")
    tar.extractall(target_directory)
    tar.close()
    os.remove(filepath)


def __download_dataset(dataset, data_urls, convert_truths, collect_image_files, datasets_directory="datasets"):
    dataset_directory = os.path.join(datasets_directory, dataset)
    os.makedirs(dataset_directory, exist_ok=True)
    for data_type, data_url in data_urls.items():
        datatype_directory = os.path.join(dataset_directory, data_type)
        images_directory = os.path.join(datatype_directory, "images")
        if os.path.isdir(images_directory):
            print("Skipping %s.%s - directory %s already exists" % (dataset, data_type, images_directory))
            continue
        print("Retrieving %s/%s..." % (dataset, data_type))
        compressed_directory = os.path.join(datasets_directory, dataset, data_type + "_compressed")
        __retrieve_tarred_content(data_url, compressed_directory)

        print("Converting %s/%s truths" % (dataset, data_type))
        truths = convert_truths(compressed_directory, data_type)
        assert truths, "No truths converted"
        truths_filepath = os.path.join(datatype_directory, "ground_truths.p")
        os.makedirs(datatype_directory)
        pickle.dump(truths, open(truths_filepath, 'wb'))

        print("Collecting %s/%s images" % (dataset, data_type))
        images_source_directory, image_files = collect_image_files(compressed_directory)
        assert image_files, "No images found"
        assert len(image_files) == len(truths), \
            "Number of images (%d) differs from truths (%d)" % (len(image_files), len(truths))
        images_truths_diffs = set(image_files) - set(truths.keys())
        truths_images_diffs = set(truths.keys()) - set(image_files)
        assert not images_truths_diffs and not truths_images_diffs, \
            "image files and truths keys differ: only in images %s | only in truths %s" % (
                ', '.join(images_truths_diffs), ', '.join(truths_images_diffs))
        os.makedirs(images_directory)
        for image_file in image_files:
            shutil.move(os.path.join(images_source_directory, image_file), os.path.join(images_directory, image_file))
        shutil.rmtree(compressed_directory)


def __collect_images(images_directory, filetype):
    images_directory = os.path.realpath(images_directory)
    image_files = glob.glob(os.path.join(images_directory, "*." + filetype))
    return images_directory, [filepath[len(images_directory) + 1:] for filepath in image_files]


def __convert_voc_truths(dataset_directory):
    assert os.path.isdir(dataset_directory)
    xml_files = glob.glob("%s/*.xml" % dataset_directory)
    truths = {}
    for file in xml_files:
        contents = xml.parse(file)
        filename = contents.find("filename").text
        object_name = contents.find("./object/name").text
        truths[filename] = object_name
    return truths


def __convert_imagenet2012_truths(dataset_directory, data_type):
    parent_directory, _ = os.path.split(dataset_directory)
    _, dataset = os.path.split(parent_directory)
    assert dataset == 'ILSVRC2012'
    assert data_type in ['train', 'val', 'test']
    annotations_path = os.path.join(parent_directory, "annotations")
    __retrieve_tarred_content("http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz", annotations_path)
    truths = {}
    with open(os.path.join(annotations_path, data_type + ".txt")) as annotations_file:
        for line in annotations_file.readlines():
            filename, truth = line.split(" ")
            truths[filename] = truth
    return truths


if __name__ == '__main__':
    # params - command line
    parser = argparse.ArgumentParser(description='Neural Net Robustness - Download Data')
    parser.add_argument('--datasets_directory', type=str, default='datasets',
                        help='The directory for the datasets')
    args = parser.parse_args()

    # weights
    __download_if_needed("http://files.heuritech.com/weights/alexnet_weights.h5", "weights/alexnet.h5")

    # datasets
    imagenet_urls = pickle.load(open(os.path.join(args.datasets_directory, 'ILSVRC2012_image_urls.p'), 'rb'))
    if not imagenet_urls:
        print("Due to copyright constraints, ILSVRC2012 image urls must not be distributed. "
              "Please pickle-dump a dictionary of the form {'train': <train_images_url.tar>, 'val': ..., 'test': ...} "
              "to datasets/ILSVRC2012_image_urls.p")
    else:
        __download_dataset("ILSVRC2012", datasets_directory=args.datasets_directory,
                           data_urls=imagenet_urls,
                           convert_truths=__convert_imagenet2012_truths,
                           collect_image_files=lambda dataset_directory: __collect_images(dataset_directory, "JPEG"))

    __download_dataset("VOC2012", datasets_directory=args.datasets_directory,
                       data_urls={
                           'train': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
                           'val': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'},
                       convert_truths=lambda dataset_directory, _:
                       __convert_voc_truths(os.path.join(dataset_directory, "VOCdevkit", "VOC2012", "Annotations")),
                       collect_image_files=lambda dataset_directory:
                       __collect_images(os.path.join(dataset_directory, "VOCdevkit", "VOC2012", "JPEGImages"), "jpg"))
