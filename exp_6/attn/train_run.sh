#!/usr/bin/sh

# Script to run several models one after another. All results will be placed in 
# experiments under model_states and metrics in directories according to the 
# name given to each model

FOLDS=4

# New Facts, DHC and SC
for fold in $(seq 0 $FOLDS)
do
    ./train.py -e ~/Results/advocate_recommendation/new/exp_2.1/cross_val/5_fold/fold_$fold/embeddings ~/Results/advocate_recommendation/new/exp_3.1/cross_val/5_fold/fold_0/embeddings \
        -en tf-idf doc2vec \
        -ed 33265 300 \
        -t ~/Datasets/DHC/variations/new/var_1/targets/case_chapters.json \
        -n ensemble_self_attn_fold_$fold \
        -p params_word2vec_200.json
done


