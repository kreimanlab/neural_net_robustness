import glob
import os
import tarfile
import xml.etree.ElementTree as xml
from urllib.request import urlretrieve
import scipy.io
import shutil


def download_if_needed(url, local_path):
    download_needed = not os.path.isfile(local_path)
    if download_needed:
        print("Downloading %s..." % local_path)
        urlretrieve(url, local_path)
    else:
        print("Skipping %s (exists already)" % local_path)
    return download_needed


def extract_tar(filepath):
    tar = tarfile.open(filepath)
    tar.extractall()
    tar.close()
    os.remove(filepath)


def convert_truths(directory, collector):
    truths = collector(directory)
    output_path = os.path.join(directory, "ground_truth.txt")
    with open(output_path, 'w') as output_file:
        for item in truths:
            output_file.write("%s\n" % item)
    return truths


def voc_truth_collector(directory):
    xmlFiles = glob.glob("%s/*.xml" % directory)
    truths = []
    for file in xmlFiles:
        contents = xml.parse(file)
        truths.append(contents.find("./object/name").text)


def imagenet_truths(directory):
    words = scipy.io.loadmat(os.path.join(directory, "synset_words.mat"))
    words = words['synset']
    words = [t[1] for t in words[:, 0]]
    return words


# weights
download_if_needed("http://files.heuritech.com/weights/alexnet_weights.h5", "alexnet_weights.h5")
# datasets
if download_if_needed("http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
                      "datasets/VOC2012_original.tar"):
    extract_tar("datasets/VOC2012_original.tar")
    os.mkdir("datasets/VOC2012")
    for file in glob.glob("datasets/VOC2012_original/JPEGImages/*.jpg"):
        shutil.move(file, os.path.join("datasets/VOC2012", file.name))
    convert_truths("datasets/VOC2012_original/Annotations", voc_truth_collector)
    os.rmdir("datasets/VOC2012_original")
