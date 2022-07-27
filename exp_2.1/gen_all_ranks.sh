#!/usr/bin/sh

FOLDS=4

# Generate similarity ranking and re-ranking for all provided FOLDS

for fold in $(seq 0 $FOLDS)
do
    python calculate_ranks.py -d ~/Datasets/DHC/variations/var_2/data/ipc_data/cross_val/20_fold/fold_$fold/adv_case_splits.json \
        -ct ~/Datasets/DHC/common/case_advs.json \
        -s ~/Results/advocate_recommendation/exp_2.1/new_cross_val/20_fold/fold_$fold/scores.json \
        -o ~/Results/advocate_recommendation/exp_2.1/new_cross_val/20_fold/fold_$fold/
done
