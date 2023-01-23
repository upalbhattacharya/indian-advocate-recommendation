#!/usr/bin/env sh

# Script to run several models one after another. All results will be placed in 
# experiments under model_states and metrics in directories according to the 
# name given to each model

# New Facts, DHC and SC
./train.py -d /DATA/upal/Datasets/DHC/variations/new/var_4/cross_val/0_fold/dropped/
    -db /DATA/upal/embeddings/statute_pred_embeddings/exp_9/ \
    -t /DATA/upal/Datasets/DHC/variations/new/var_4/targets/dropped/case_advs.json \
    -n adv_preds \
    -p params_word2vec_200.json \
    -ul /DATA/upal/Datasets/DHC/variations/new/var_4/targets/dropped/selected_advs.txt \
    -id 0


