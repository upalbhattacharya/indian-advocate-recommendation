#!/usr/bin/env sh

FOLDS=4

for fold in $(seq 0 $FOLDS)
do
    ./cosine_calc.py -d /home/workboots/Repos/my_repos/advocate_recommendation/exp_6/attn/data/ \
        -q /home/workboots/Repos/my_repos/advocate_recommendation/exp_6/attn/data/ \
        -o . 
done

