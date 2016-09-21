
## Setup
This project requires Python version >= 3.2.

First, install library requirements by running `pip install -r requirements.txt`.

Then download model weights and datasets by running `python download_data.py`. This will:
* download the following weights
  * [alexnet](http://files.heuritech.com/weights/alexnet_weights.h5)
* download the following datasets
  * [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)
  * [ILSVRC2012](http://image-net.org/challenges/LSVRC/2012/)


## Running
Use `python run.py` to re-train and evaluate models, 
`python perturb_weights.py` to perturb the weights, 
`python analyze.py` to analyze weights and results.

### Re-Training
Run `python run.py` with the `--weights` argument not set to re-train the model on the dataset(s), for instance:

    python run.py --model alexnet --dataset VOC2012
will re-train the model `alexnet` on the dataset `VOC2012`.
Note that the model will not be trained from scratch but instead start with its associated weights, in the example `weights/alexnet.h5`.

### Evaluation
Run `python run.py` with the `--weights` argument set to evaluate the model on the dataset(s), for instance:

    python run.py --model alexnet --weights alexnet_retrained_on_VOC2012 --dataset VOC2012
will evaluate the model `alexnet` with the weights `weights/alexnet_retrained_on_VOC2012.h5` on the dataset `VOC2012`.

### Weight Perturbation
Run `python perturb_weights.py` to perturb the weights of a model.
For instance:

    python perturb_weights.py --weights alexnet --layer conv_1 conv_2 conv_3 conv_4 conv_5 dense_1 dense_2 dense_3 --ratio 0.1 0.2 0.3 0.4 0.5 --num_perturbations 5
will perturb all 8 layers of `weights/alexnet.h5` using the ratios `{0.1, 0.2, ..., 0.5}` and 5 different random variations for each layer.

### Analysis
Run `python analyze.py <task>` to analyze the results 
where `<task>` is one of `num_weights`, `weight_diffs`, `performances`.

For the `performances` task, if a given weight file itself does not exist, 
but variations of it with appended `-num[0-9]+` etc. do exist, 
then those will be used and the results averaged across these variations.
For instance, if `--weights alexnet-conv1-draw0.5` is provided 
and the results for these weights do not exist, the script wil search for 
`alexnet-conv1-draw0.5-num1.p`, `alexnet-conv1-draw0.5-num2.p` etc. instead.

## Running with LSF
To run any of the Python progams on a cluster with [LSF](https://www.ibm.com/support/knowledgecenter/SSETD4_9.1.2/lsf_kc_cmd_ref.html),  
use the `run_lsf.sh` script, for instance:

    ./run_lsf.sh run.py --model alexnet
Note that this script is tailored to the Orchestra cluster at the Harvard Medical School, 
specifically a queue named `gpu` is assumed to exist. 