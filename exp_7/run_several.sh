#!/usr/bin/sh
# Script to run several models one after another. All results will be placed in 
# experiments under model_states and metrics in directories according to the 
# name given to each model

fold=4

python train.py -dtr ~/Results/advocate_recommendation/new/exp_2.1/cross_val/5_fold/fold_$fold/embeddings/train_rep \
    -dts ~/Results/advocate_recommendation/new/exp_2.1/cross_val/5_fold/fold_$fold/embeddings/test_rep \
    -t ~/Datasets/DHC/variations/new/var_1/targets/case_chapters.json \
    -n tf_idf_embed_pred_fold_$fold \
    -p params_word2vec_200_tf_idf.json


