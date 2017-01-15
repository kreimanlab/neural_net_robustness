
## Setup
This project requires Python version >= 3.2.

First, install library requirements by running `pip install -r requirements.txt`.

Then download model weights and datasets by running `python download_data.py`. This will:
* download the following weights, each pre-trained on an ILSVRC training set
  * Alexnet: ILSVRC2012, [convnetskeras](https://github.com/heuritech/convnets-keras)' [model files](http://files.heuritech.com/weights/alexnet_weights.h5)
  * VGG16 and VGG19: ILSVRC2014, [keras](https://github.com/fchollet/deep-learning-models) uses [VGG](http://www.robots.ox.ac.uk/~vgg/research/very_deep/)'s weights ([16](http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel), [19](http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel))
  * ResNet50: ILSVRC2015,  [keras](https://github.com/fchollet/deep-learning-models) uses [Kaiming He](https://github.com/KaimingHe/deep-residual-networks)'s weights ([model files](https://onedrive.live.com/?authkey=%21AAFW2-FVoxeVRck&id=4006CBB8476FF777%2117887&cid=4006CBB8476FF777))
  * Inceptionv3: [keras](https://github.com/fchollet/deep-learning-models)' trained weights
* download the following datasets
  * [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) (training and validation data)
  * [ILSVRC2012](http://image-net.org/challenges/LSVRC/2012/) (training, validation and test data - note that the labels for the test data are all zero)


## Running
Use `python run.py` to re-train models and predict datasets, 
`python perturb_weights.py` to perturb the weights, 
`python analyze.py` to analyze weights and results.

### Re-Training
Run `python run.py` with the `--weights` argument not set to re-train the model on the dataset(s), for instance:

    python run.py --model alexnet --datasets VOC2012/val
will re-train the model `alexnet` on the dataset `VOC2012/val`.
Note that the model will not be trained from scratch but instead start with its associated weights, in the example `weights/alexnet.h5`.

### Prediction
Run `python run.py` with the `--weights` argument set to predict the dataset(s) with the model, for instance:

    python run.py --model alexnet --weights alexnet_retrained_on_VOC2012 --datasets VOC2012/val
will predict the dataset `VOC2012/val` with the model `alexnet` using the weights `weights/alexnet_retrained_on_VOC2012.h5`.

### Weight Perturbation
Run `python perturb_weights.py` to perturb the weights of a model.
For instance:

    python perturb_weights.py --weights alexnet --layer conv_1 conv_2 conv_3 conv_4 conv_5 dense_1 dense_2 dense_3 --ratio 0.1 0.2 0.3 0.4 0.5 --num_perturbations 5
will perturb all 8 layers of `weights/alexnet.h5` using the ratios `{0.1, 0.2, ..., 0.5}` and 5 different random variations for each layer.

### Analyze Predictions
Run `python analyze.py` to analyze the results.
For instance:

    python run.py --model alexnet --weights perturbations/alexnet-conv_1-draw0.10 --datasets ILSVRC2012/val --metrics top5error
will analyze the predictions produced by the model `alexnet` 
with weights `weights/perturbations/alexnet-conv_1-draw0.10.h5` 
on `ILSVRC2012/val` using the `top5error` metric.

### Plotting
Run `python plot.py <task>` to plot weights 
or the results produced by a previous analysis
where `<task>` is one of `num_weights`, `weight_diffs`, `performances`.

#### Weight Inference
If a results file for some weights itself does not exist, 
but variations of it with appended `-num[0-9]+` etc. do exist, 
then those will be used and the results averaged across these variations.
For instance, if `--weights alexnet-conv1-draw0.5` is provided 
and the results for these weights do not exist, the script wil search for 
`alexnet-conv1-draw0.5-num1.p`, `alexnet-conv1-draw0.5-num2.p` etc. instead.


## Running with LSF
To run any of the Python progams on a cluster with [LSF](https://www.ibm.com/support/knowledgecenter/SSETD4_9.1.2/lsf_kc_cmd_ref.html),  
use the `run_lsf.sh` script, for instance:

    ./run_lsf.sh run.py --model alexnet
Note that this script is tailored to our own cluster, 
specifically a queue named `gpu` is assumed to exist.
