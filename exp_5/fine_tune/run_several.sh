#!/usr/bin/sh

FOLDS=4
# Script to run several models one after another. All results will be placed in 
# experiments under model_states and metrics in directories according to the 
# name given to each model

# New Facts, DHC and SC

for fold in $(seq 0 $FOLDS)
do
    python train.py -d ~/Datasets/DHC/variations/var_3/data/ipc_data/cross_val/5_fold/fold_$fold/ \
        -t ~/Datasets/DHC/variations/var_3/targets/ipc_case_offences.json \
        -n SBERT_distilroberta_fold_$fold -p params_distilroberta.json 
done


