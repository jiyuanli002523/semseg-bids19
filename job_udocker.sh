#!/bin/bash
##### INFO ######
# This script supposes to:
# 1. run 2dsemseg2019 code inside the container by means of udocker
#
# VKozlov @20-Oct-2018
#
# udocker: https://github.com/indigo-dc/udocker
#
################

####### MAIN CONFIG #######
UCONTAINER="bids2019-gpu"                             # container to use
NUMGPUS=4                                             # in some systems a node has >1 GPU. e.g. in LSDF one can set NUMGPUS=4 (max)
HOSTData=$HOME/datasets/vaihingen/data
HOSTModels=$HOME/datasets/vaihingen/models

DLScript="/2dsemseg/2dsemseg/train_resnet50_fcn.py"   # Deep learning script to run (path inside container)
RUNName="udocker-gpus=$NUMGPUS"
DLData="/2dsemseg/data"
DLModel=/2desemseg/models/$RUNName"_weights.hdf5"
DLCsvLog=/2dsemseg/models/$RUNName"-log.csv"

DLScriptOpts="--data_path=$DLData --model=$DLModel --log=$DLCsvLog --n_gpus=$NUMGPUS"      # options for the script

#--------------------------
UDOCKER_DIR="$HOME/.udocker"                          # udocker main directory.
###########################
echo "==================================="
echo "=> udocker container: $UCONTAINER"
echo "==================================="
### following line has to be run only first time
udocker setup --nvidia ${UCONTAINER}
##### >>> now run the script:
udocker run -v $HOSTData:/2dsemseg/data -v $HOSTModels:/2dsemseg/models ${UCONTAINER} python $DLScript $DLScriptOpts
