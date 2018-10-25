2D Semantic Segmentation
==============================

2D semantic segmentation (Vaihingen dataset). 
Version fixed for the [BiDS 2019](https://www.bigdatafromspace2019.org/QuickEventWebsitePortal/2019-conference-on-big-data-from-space-bids19/bids-2019) Conference.

## Code running

You can either use python's virtual environment or use Docker image to run the code.

### Using Virtual Environment
In our tests for [BiDS 2019](https://www.bigdatafromspace2019.org/QuickEventWebsitePortal/2019-conference-on-big-data-from-space-bids19/bids-2019) Conference we used:
1. python 2.7
2. [CUDA Toolkit 9.0.176](https://developer.nvidia.com/cuda-90-download-archive) and [cuDNN 7.0.5](https://developer.nvidia.com/rdp/cudnn-archive)
3. If you do not have virtualenv, please, install it. For example:
```
$ curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
$ sudo python get-pip.py
$ sudo pip install -U virtualenv
```

4. virtual environment is created by 
```
$ virtualenv -p python2.7 $HOME/.venv/bids2019
```
5. activate the virtual environment: 
```
$ source $HOME/.venv/bids2019/bin/activate
```
6. clone the code from this repository to your local directory

7. go to this directory, install required python packages listed in requirements.txt file:
```
(bids2019)$ pip install -r requirements.txt
```

### Using (u)Docker
We also provide the code and all necessary dependencies in the Docker image at Docker Hub:
https://hub.docker.com/r/vykozlov/2dsemseg/tags/ , tag 'bids2019'

In the paper we use _uDocker_ container tool from [udocker/devel branch](https://github.com/indigo-dc/udocker/tree/devel) which has NVIDIA support (`--nvidia flag`). In order to run the code:
1. Install _uDocker_, refer to [udocker/installation manual](https://github.com/indigo-dc/udocker/blob/devel/doc/installation_manual.md) for more details but in short:
  a) best go to one of your $PATH directories, e.g. `$HOME/.local/bin` (depends on your system, type `echo $PATH` to check!)
  b) then
```
$ curl https://raw.githubusercontent.com/indigo-dc/udocker/devel/udocker.py > udocker
$ chmod u+rx ./udocker
$ export UDOCKER_DIR=$HOME/.udocker
$ ./udocker install
```
2. Use `$ udocker pull vykozlov/2dsemseg:bids2019` to pull Docker image locally
3. `$ udocker create --name=bids2019 vykozlov/2dsemseg:bids2019` to create local container
4. If your host system has GPUs, run
```
$ udocker setup --execmode=F3 --nvidia bids2019
```
5. `$ udocker run -v $HOSTDIR_WITH_DATA:/2dsemseg/data -v $HOSTDIR_FOR_MODELS:/2dsemseg/models` 

## Project Organization

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Placeholder to put training and validation data
    │
    ├── docker             <- Directory for Dockerfile(s)
    │
    ├── models             <- Placeholder for trained and serialized models, model predictions, or model summaries
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── 2dsemseg           <- Source code for use in this project.
        ├── __init__.py    <- Makes 2dsemseg a Python module
        │
        ├── augmentation.py  <- to apply augmentation on original data
        │
        ├── data_io.py       <- to generate the Training and Validation Set from original data
        │
        ├── evaluate_network.py    <- to test trained netowrk
        │
        └── model_generator.py     <- model generator :-)
        │
        └── resnet_edit.py         <- ResNet50 for Keras
        │
        └── storeincsv.py          <- module to write csv summary
        │
        └── train_resnet50_fcn.py  <- main code for training


