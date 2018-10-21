#!/bin/bash
##### INFO ######
# Script to run container with 2dsemseg code by means of udocker
#
# udocker: https://github.com/indigo-dc/udocker
#
# VKozlov @20-Oct-2018
################

####### MAIN CONFIG #######
# NUMGPUS: number of GPUs to use, up to the max number available on ONE node
NUMGPUS=1
# UCONTAINER: udocker container to use
# Example to prepare the container:
# $ udocker pull vykozlov/2dsemseg:bids2019-gpu
# $ udocker create --name=bids2019-gpu vykozlov/2dsemseg:bids2019-gpu
UCONTAINER="bids2019-gpu"

# RUNName: name of the run
RUNName="udocker-gpus=$NUMGPUS"

HOSTData=$HOME/datasets/vaihingen/data                # HOST directory with Training and Validation data
HOSTModels=$HOME/datasets/vaihingen/models            # HOST directory for the output model with weights

DLScript="/2dsemseg/2dsemseg/train_resnet50_fcn.py"   # deep learning script to run (path inside container. fixed in the container)
DLData=/2dsemseg/data                                 # mount point inside container for Training and Validation data
DLModels=/2desemseg/models                            # mount point inside container for the output model

# DLScriptOpts: options for the script. N.B. default n_epochs=20
DLScriptOpts="--data_path=${DLData} --model=${DLModels}/${RUNName}_weights.hdf5 --log=${DLModels}/${RUNName}_log.csv --n_gpus=${NUMGPUS}"

###########################
UDOCKER_DIR="$HOME/.udocker"                          # udocker main directory.
echo "[INFO] udocker container: $UCONTAINER"
# following line has to be run only first time
udocker setup --nvidia ${UCONTAINER}
##### >>> now run the script:
# we mount HOST directory with Training and Validation data 
# and directory for the output model inside the container
udocker run -v $HOSTData:$DLData -v $HOSTModels:$DLModels ${UCONTAINER} python $DLScript $DLScriptOpts
