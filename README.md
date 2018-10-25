2D Semantic Segmentation
==============================

2D semantic segmentation ([Vaihingen dataset](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-vaihingen.html)). 
Version fixed for the [BiDS 2019](https://www.bigdatafromspace2019.org/QuickEventWebsitePortal/2019-conference-on-big-data-from-space-bids19/bids-2019) Conference.

## Code running

You can either use python's virtual environment or use Docker image to run the code. We recommend the latter way.

### Using (u)Docker
The code and all necessary dependencies are provided in the Docker image at Docker Hub:
https://hub.docker.com/r/vykozlov/semseg/tags/ , tag 'bids19-gpu'

#### Pre-requisites
In the paper for [BiDS 2019](https://www.bigdatafromspace2019.org/QuickEventWebsitePortal/2019-conference-on-big-data-from-space-bids19/bids-2019) Conference we use _uDocker_ container tool from [udocker/devel branch](https://github.com/indigo-dc/udocker/tree/devel) which has NVIDIA support (`--nvidia flag`). Please, notice that _uDocker_ is entirely a user tool, i.e. **no** root priveleges of any kind are needed.
1. Install _uDocker_, refer to [udocker/installation manual](https://github.com/indigo-dc/udocker/blob/devel/doc/installation_manual.md) for more details but in short:
  - best go to one of your $PATH directories, e.g. `$HOME/.local/bin` (depends on your system, type `echo $PATH` to check!). Then
  ```
  $ curl https://raw.githubusercontent.com/indigo-dc/udocker/devel/udocker.py > udocker
  $ chmod u+rx ./udocker
  $ export UDOCKER_DIR=$HOME/.udocker
  $ ./udocker install
  ```
2. `$ udocker pull vykozlov/semseg:bids19-gpu` to pull Docker image locally
3. `$ udocker create --name=bids19-gpu vykozlov/semseg:bids19-gpu` to create local container
4. `$ udocker setup --execmode=F3 --nvidia bids19-gpu` to enable Fakechroot execution mode and to use the host NVIDIA driver

#### Prepare data
1. Download Vaihingen dataset in $HOSTDIR_WITH_DATA/raw
2. Prepare data for training:
```
$ udocker run -v $HOSTDIR_WITH_DATA:/semseg-bids19/data bids19-gpu python /semseg-bids19/semseg/data_io.py /semseg-bids19/data/raw /semseg-bids19/data
```
where 
  * $HOSTDIR_WITH_DATA : directory to put resulting vaihingen_train.hdf5 and vaihingen_val.hdf5 files. $HOSTDIR_WITH_DATA/raw is expected to have _raw_ .hdf5 files, i.e. which you downloaded (see above).

#### Run training
```
$ udocker run -v $HOSTDIR_WITH_DATA:/semseg-bids19/data -v $HOSTDIR_FOR_MODELS:/semseg-bids19/models bids19-gpu
```
where 
  * $HOSTDIR_WITH_DATA : directory at your host with vaihingen .hdf5 files prepared for training
  * $HOSTDIR_FOR_MODELS: directory at your host where output training files will be stored.

By default this will run the followinig command inside container using 20 epochs for training:
```
python /semseg-bids19/semseg/train_resnet50_fcn.py \
       --data_path=/semseg-bids19/data \
       --model=/semseg-bids19/models/resnet50_fcn_weights.hdf5 \
       --log=/semseg-bids19/models/resnet50_fcn_weights_log.csv
```
If you want to redefine `train_resnet50_fcn.py` parameters, your run for example:
```
$ udocker run -v $HOSTDIR_WITH_DATA:/semseg-bids19/data -v $HOSTDIR_FOR_MODELS:/semseg-bids19/models bids19-gpu python /semseg-bids19/semseg/train_resnet50_fcn.py --data_path=/semseg-bids19/data --model=/semseg-bids19/models/resnet50_fcn_weights.hdf5 --log=/semseg-bids19/models/resnet50_fcn_weights_log.csv --n_epochs=25
```
**Best way** is to put this in a shell script. For the example, please, see `job_udocker.sh`

#### Usage of training_resnet50_fcn.py
```
usage: train_resnet50_fcn.py [-h] [--data_path DATA_PATH] [--model MODEL]
                             [--n_epochs N_EPOCHS] [--n_gpus N_GPUS]
                             [--no_augmentation] [--load_weights] [--log LOG]
```

optional arguments:
```
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Location of vaihingen_train.hdf5 and
                        vaihingen_val.hdf5 (e.g. /homea/hpclab/train002/semseg/data )
  --model MODEL         Location + name of the output model 
                        (e.g., /homea/hpclab/train002/semseg/models/resnet50_fcn_weights.hdf5)
  --n_epochs N_EPOCHS   Number of epochs to train on
  --n_gpus N_GPUS       Number of GPUs to train on (one node only!)
  --no_augmentation     Skip augmentation
  --load_weights        Use transfer learning and load pre-trained weights
  --log LOG             Location + name of the csv log file
```

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
$ virtualenv -p python2.7 $HOME/.venv/bids19
```
5. activate the virtual environment: 
```
$ source $HOME/.venv/bids19/bin/activate
```
6. clone the code from this repository to your local directory

7. go to this directory, install required python packages listed in requirements.txt file:
```
(bids19)$ pip install -r requirements.txt
```
In the following `(bids19)$` indicates that you have to act from the virtual environment.

#### Prepare data
Similar to udocker:
1. Download Vaihingen dataset in $HOSTDIR_WITH_DATA/raw
2. Prepare data for training:
```
(bids19)$ python ./data_io.py $HOSTDIR_WITH_DATA/raw $HOSTDIR_WITH_DATA
```
where 
  * $HOSTDIR_WITH_DATA : directory to put resulting vaihingen_train.hdf5 and vaihingen_val.hdf5 files. 

#### Run training
```
(bids19)$ python ./train_resnet50_fcn.py --data_path=$HOSTDIR_WITH_DATA --model=$HOSTDIR_FOR_MODELS/resnet50_fcn_weights.hdf5 --log=$HOSTDIR_FOR_MODELS/resnet50_fcn_weights_log.csv --n_epochs=25
```
Again **best way** would be to put this in a shell script. For the example, please, see `job_bmetal.sh`.

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
    ├── semseg           <- Source code for use in this project.
        ├── __init__.py    <- Makes semseg a Python module
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


