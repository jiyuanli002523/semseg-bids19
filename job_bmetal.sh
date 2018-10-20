#!/bin/bash

##### INFO #####################
# script to be submitted to the batch system as a job
# bare-metal case
################################

####### MAIN CONFIG #######
NUMGPUS=4
VIRTEnv="bids2019"
source $HOME/.venv/$VIRTEnv/bin/activate

DLProject=$HOME/workspace/2dsemseg
DLScript=$DLProject/2dsemseg/train_resnet50_fcn.py
RUNName="bmetal-$VIRTEnv-gpus=$NUMGPUS"
DLData=$HOME/datasets/vaihingen/data
DLModel=$HOME/datasets/vaihingen/models/$RUNName"_weights.hdf5"
DLLog=$DLProject/reports

DLScriptOpts="--data_path=$DLData --model=$DLModel --log=$DLLog/$RUNName-log.csv --n_gpus=$NUMGPUS"
########

## for NFS v2/v3 local cache path needed
export CUDA_CACHE_PATH=/tmp
#
## libraries for cuda
export CUDASYS='/usr/local/cuda'
export CUDNN=$PROJECT'/.local/lib/cuda'
export PATH=$CUDASYS'/bin':$PATH
export PATH=$CUDASYS'/libnvvp':$PATH
export PATH=$CUDASYS'/libnsight':$PATH
export LD_LIBRARY_PATH=$CUDASYS'/lib64':$CUDASYS'/extras/CUPTI/lib64':$CUDNN/lib64:$LD_LIBRARY_PATH
##
## >>> now run the script:
python $DLScript $DLScriptOpts
