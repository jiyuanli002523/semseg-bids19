2D Semantic Segmentation
==============================

2D semantic segmentation (Vaihingen dataset). 
Version fixed for the [BiDS 2019](https://www.bigdatafromspace2019.org/QuickEventWebsitePortal/2019-conference-on-big-data-from-space-bids19/bids-2019) Conference.

## Code running

You can either use python's virtual environment or use Docker image to run the code. We recommend the latter way.

### Using (u)Docker
The code and all necessary dependencies are provided in the Docker image at Docker Hub:
https://hub.docker.com/r/vykozlov/2dsemseg/tags/ , tag 'bids2019-gpu'

#### Pre-requisites
In the paper for [BiDS 2019](https://www.bigdatafromspace2019.org/QuickEventWebsitePortal/2019-conference-on-big-data-from-space-bids19/bids-2019) Conference we use _uDocker_ container tool from [udocker/devel branch](https://github.com/indigo-dc/udocker/tree/devel) which has NVIDIA support (`--nvidia flag`). Please, notice that _uDocker_ is entirely user tool, i.e. **no** root priveleges of any kind are needed.
1. Install _uDocker_, refer to [udocker/installation manual](https://github.com/indigo-dc/udocker/blob/devel/doc/installation_manual.md) for more details but in short:
  * best go to one of your $PATH directories, e.g. `$HOME/.local/bin` (depends on your system, type `echo $PATH` to check!). Then
  ```
  $ curl https://raw.githubusercontent.com/indigo-dc/udocker/devel/udocker.py > udocker
  $ chmod u+rx ./udocker
  $ export UDOCKER_DIR=$HOME/.udocker
  $ ./udocker install
  ```
2. `$ udocker pull vykozlov/2dsemseg:bids2019` to pull Docker image locally
3. `$ udocker create --name=bids2019-gpu vykozlov/2dsemseg:bids2019` to create local container
4. `$ udocker setup --execmode=F3 --nvidia bids2019-gpu` to enable Fakechroot execution mode and to use the host NVIDIA driver

#### Prepare data
1. Download Vaihingen dataset in $HOSTDIR_WITH_DATA/raw
2. Prepare data for training:
```
$ udocker run -v $HOSTDIR_WITH_DATA:/2dsemseg/data bids2019 python /2dsemseg/2dsemseg/data_io.py /2dsemseg/data/raw /2dsemseg/data
```
where 
  * $HOSTDIR_WITH_DATA : directory to put resulting vaihingen_train.hdf5 and vaihingen_val.hdf5 files. 

#### Run training
`$ udocker run -v $HOSTDIR_WITH_DATA:/2dsemseg/data -v $HOSTDIR_FOR_MODELS:/2dsemseg/models bids2019`
where 
  * $HOSTDIR_WITH_DATA : directory at your host with Vaihingen .hdf5 files
  * $HOSTDIR_FOR_MODELS: directory at your host where output training files will be stored

By default this will run the followinig command inside container using 20 epochs for training:
```
python /2dsemseg/2dsemseg/train_resnet50_fcn.py \
       --data_path=/2dsemseg/data \
       --model=/2dsemseg/models/resnet50_fcn_weights.hdf5 \
       --log=/2dsemseg/models/resnet50_fcn_weights_log.csv
```
If you want to redefine `train_resnet50_fcn.py` parameters, your run for example:
```
$ udocker run -v $HOSTDIR_WITH_DATA:/2dsemseg/data -v $HOSTDIR_FOR_MODELS:/2dsemseg/models bids2019 python /2dsemseg/2dsemseg/train_resnet50_fcn.py --data_path=/2dsemseg/data --model=/2dsemseg/models/resnet50_fcn_weights.hdf5 --log=/2dsemseg/models/resnet50_fcn_weights_log.csv --n_epochs=25
```
**Best way** is to put this in a shell script. For the example, please, see `job_udocker.sh`

### Using Virtual Environment
#### Pre-requisites
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
In the following `(bids2019)$` indicates that you have to act from the virtual environment.

#### Prepare data
Similar to udocker:
1. Download Vaihingen dataset in $HOSTDIR_WITH_DATA/raw
2. Prepare data for training:
```
(bids2019)$ python ./data_io.py $HOSTDIR_WITH_DATA/raw $HOSTDIR_WITH_DATA
```
where 
  * $HOSTDIR_WITH_DATA : directory to put resulting vaihingen_train.hdf5 and vaihingen_val.hdf5 files. 

#### Run training
```
(bids2019)$ python ./train_resnet50_fcn.py --data_path=$HOSTDIR_WITH_DATA --model=$HOSTDIR_FOR_MODELS/resnet50_fcn_weights.hdf5 --log=$HOSTDIR_FOR_MODELS/resnet50_fcn_weights_log.csv --n_epochs=25
```
Again **best way** would be to put this in a shell script. For the example, please, see `job_bmetal.sh`

If you have to submit your job to a batch system, you can use the script, either `job_udocker.sh` or `job_bmetal.sh`, in your job submission. Please, adjust scripts to your needs :-)

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


