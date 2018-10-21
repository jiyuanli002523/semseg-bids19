#!/bin/bash

##### INFO #####################
# Script to be submitted to the batch system as a job
# bare-metal case
################################

####### MAIN CONFIG #######
# NUMGPUS: number of GPUs to use, up to the max number available on ONE node
NUMGPUS=4
# VIRTEnv: python virtual environment where all necessary packages are installed
VIRTEnv="bids2019"
source $HOME/.venv/$VIRTEnv/bin/activate

# RUNName: name of the run
RUNName="bmetal-$VIRTEnv-gpus=$NUMGPUS"

DLProject=$HOME/workspace/2dsemseg                     # directory with the source code
DLScript=$DLProject/2dsemseg/train_resnet50_fcn.py     # deep learning script to run
DLData=$HOME/datasets/vaihingen/data                   # directory with Training and Valiation data (.hdf5)
DLModels=$HOME/datasets/vaihingen/models               # directory for the output model file with weights

# DLScriptOpts: options for the script. N.B. default n_epochs is 20
DLScriptOpts="--data_path=$DLData --model=$DLModels/$RUNName'_weights.hdf5' --log=$DLModels/$RUNName'_log.csv' --n_gpus=$NUMGPUS"
########

## for NFS v2/v3 local cache path needed
export CUDA_CACHE_PATH=/tmp
## libraries for cuda
export CUDASYS='/usr/local/cuda'
export PATH=$CUDASYS'/bin':$PATH
export PATH=$CUDASYS'/libnvvp':$PATH
export PATH=$CUDASYS'/libnsight':$PATH
export LD_LIBRARY_PATH=$CUDASYS'/lib64':$CUDASYS'/extras/CUPTI/lib64':$LD_LIBRARY_PATH
##
## >>> now run the script:
python $DLScript $DLScriptOpts
