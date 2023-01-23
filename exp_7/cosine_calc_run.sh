#!/usr/bin/env sh

FOLDS=4

for fold in $(seq 0 $FOLDS)
do
    ./cosine_calc.py -d /home/workboots/Results/advocate_recommendation/new/exp_7/tfidf/cross_val/5_fold/fold_$fold/embeddings/train/\
    -q /home/workboots/Results/advocate_recommendation/new/exp_7/tfidf/cross_val/5_fold/fold_$fold/embeddings/test\
        -o /home/workboots/Results/advocate_recommendation/new/exp_7/tfidf/cross_val/5_fold/fold_$fold/results/ \
        -l /home/workboots/Results/advocate_recommendation/new/exp_7/tfidf/cross_val/5_fold/fold_$fold/logs/
done

