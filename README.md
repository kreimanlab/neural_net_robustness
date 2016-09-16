
## Setup
First, install library requirements by running `pip install -r requirements.txt`.

Then download model weights and datasets by running `python setup.py`. This will:
* download the following weights
  * [alexnet](http://files.heuritech.com/weights/alexnet_weights.h5)
* download the following datasets
  * [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)


## Running
Use `python run.py` to re-train and evaluate models.

### Re-Training
If the `--weights` argument is not set, the model will be trained on the dataset(s), for instance:

    python run.py --model alexnet --dataset VOC2012
will re-train the model `alexnet` on the dataset `VOC2012`.
Note that the model will not be trained from scratch but instead start with its associated weights, in the example `weights/alexnet.h5`.

### Evaluation
If the `--weights` argument is set, the model will be evaluated on the dataset(s), for instance:

    python run.py --model alexnet --weights alexnet_retrained_on_VOC2012 --dataset VOC2012
will evaluate the model `alexnet` with the weights `weights/alexnet_retrained_on_VOC2012.h5` on the dataset `VOC2012`.
