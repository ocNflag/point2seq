# Point2Seq

This is a reproduced repo of Point2Seq for 3D object detection. 

The code is mainly based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

## Introduction
We provide code and training configurations of PointSeq on the ONCE and Waymo Open dataset. Checkpoints will not be released.  


## Requirements
The codes are tested in the following environment:
* Ubuntu 18.04
* Python 3.6
* PyTorch 1.5
* CUDA 10.1
* OpenPCDet v0.3.0
* spconv v1.2.1

## Installation
a. Clone this repository.

b. Install the dependent libraries as follows:

* Install the dependent python libraries: 
```
pip install -r requirements.txt 
```

* Install the SparseConv library, we use the implementation from [`[spconv]`](https://github.com/traveller59/spconv). 
    * If you use PyTorch 1.1, then make sure you install the `spconv v1.0` with ([commit 8da6f96](https://github.com/traveller59/spconv/tree/8da6f967fb9a054d8870c3515b1b44eca2103634)) instead of the latest one.
    * If you use PyTorch 1.3+, then you need to install the `spconv v1.2`. As mentioned by the author of [`spconv`](https://github.com/traveller59/spconv), you need to use their docker if you use PyTorch 1.4+. 

c. Compile CUDA operators by running the following command:
```shell
python setup.py develop
```

## Training

All the models are trained with Tesla V100 GPUs (32G). 
If you use different number of GPUs for training, it's necessary to change the respective training epochs to attain a decent performance.
