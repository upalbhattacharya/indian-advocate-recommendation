#!/usr/bin/env sh

FOLDS=4

for fold in $(seq 0 $FOLDS)
do
    ./bm25.py -f ~/Datasets/DHC/variations/var_2/data/ipc_data/fact_sentences/ \
        -d ~/Datasets/DHC/variations/var_2/data/ipc_data/cross_val/20_fold/fold_$fold/adv_case_splits.json \
        -o ~/Results/advocate_recommendation/exp_2.1/new_cross_val/20_fold/fold_$fold/
done
