#!/bin/bash

##### INFO #####################
# script to be submitted to the batch system as a job
# bare-metal case
################################

####### MAIN CONFIG #######
# NUMGPUS: number of GPUs to use, up to max number of cards available at ONE node
NUMGPUS=4
# VIRTEnv: python virtual environment where all necessary packages are installed
VIRTEnv="bids2019"
source $HOME/.venv/$VIRTEnv/bin/activate

DLProject=$HOME/workspace/2dsemseg                                 # directory with the source code
DLScript=$DLProject/2dsemseg/train_resnet50_fcn.py                 # deep learning script to run
RUNName="bmetal-$VIRTEnv-gpus=$NUMGPUS"                            # name of the run
DLData=$HOME/datasets/vaihingen/data                               # directory with training and valiation data (.hdf5)
DLModel=$HOME/datasets/vaihingen/models/$RUNName"_weights.hdf5"    # path to output model file with weights
DLLog=$DLProject/reports                                           # where to store summary report

DLScriptOpts="--data_path=$DLData --model=$DLModel --log=$DLLog/$RUNName-log.csv --n_gpus=$NUMGPUS --n_epochs=20"
########

## for NFS v2/v3 local cache path needed
export CUDA_CACHE_PATH=/tmp
#
## libraries for cuda
export CUDASYS='/usr/local/cuda'
export PATH=$CUDASYS'/bin':$PATH
export PATH=$CUDASYS'/libnvvp':$PATH
export PATH=$CUDASYS'/libnsight':$PATH
export LD_LIBRARY_PATH=$CUDASYS'/lib64':$CUDASYS'/extras/CUPTI/lib64':$LD_LIBRARY_PATH
##
## >>> now run the script:
python $DLScript $DLScriptOpts
