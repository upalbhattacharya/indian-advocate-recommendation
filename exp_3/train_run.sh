#!/usr/bin/env sh

FOLDS=4

for fold in $(seq 0 $FOLDS)
do
    ./train.py -f /home/workboots/Datasets/DHC/common_new/preprocess/fact_sentences/ \
        -d /home/workboots/Datasets/DHC/variations/new/var_1/cross_val/5_fold/fold_$fold/adv_split_info.json \
        -e 30 \
        -o /home/workboots/Results/advocate_recommendation/new/exp_3/cross_val/5_fold/fold_$fold/ \
        -l /home/workboots/Results/advocate_recommendation/new/exp_3/cross_val/5_fold/fold_$fold/logs/
done
