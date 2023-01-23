#!/usr/bin/env sh

FOLDS=4

for fold in $(seq 0 $FOLDS)
do
    ./bm25.py -f ~/Datasets/DHC/common/preprocess/fact_sentences/ \
        -d ~/Datasets/DHC/variations/new/var_1/cross_val/5_fold/fold_$fold/adv_split_info.json \
        -o ~/Results/advocate_recommendation/new/exp_2/cross_val/5_fold/fold_$fold/ \
        -l ~/Results/advocate_recommendation/new/exp_2/cross_val/5_fold/fold_$fold/logs/
done
