#!/usr/bin/sh

FOLDS=4

# Script to run several models one after another. All results will be placed in 
# experiments under model_states and metrics in directories according to the 
# name given to each model

# New Facts, DHC and SC
for fold in $(seq 0 $FOLDS)
do
    ./generate_embeddings.py -e ~/Results/advocate_recommendation/new/exp_2.1/cross_val/5_fold/fold_$fold/embeddings/test_rep/ ~/Results/advocate_recommendation/new/exp_3.1/cross_val/5_fold/fold_0/embeddings/test_rep/ \
        -en tf-idf doc2vec \
        -ed 33265 300 \
        -t ~/Datasets/DHC/variations/new/var_1/targets/case_chapters.json \
        -n ensemble_self_attn_generate_fold_$fold -p params_word2vec_200.json \
        -r ~/Repos/my_repos/advocate_recommendation/exp_6/attn/experiments/model_states/ensemble_self_attn_fold_0/best.pth.tar \
        -s ~/Repos/my_repos/advocate_recommendation/exp_6/attn/data
done



