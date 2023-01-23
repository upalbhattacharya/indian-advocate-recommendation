#!/usr/bin/env sh

FOLDS=4

for fold in $(seq 0 $FOLDS)
do
    ./make_rep.py -f /home/workboots/Datasets/DHC/common_new/preprocess/fact_sentences/ \
        -d /home/workboots/Datasets/DHC/variations/new/var_1/cross_val/5_fold/fold_$fold/adv_split_info.json \
        -m /home/workboots/Results/advocate_recommendation/new/exp_3.1/cross_val/5_fold/fold_$fold/model \
        -o /home/workboots/Results/advocate_recommendation/new/exp_3.1/cross_val/5_fold/fold_$fold/embeddings \
        -l /home/workboots/Results/advocate_recommendation/new/exp_3.1/cross_val/5_fold/fold_$fold/logs
done

