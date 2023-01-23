#!/usr/bin/sh

# Script to run several models one after another. All results will be placed in 
# experiments under model_states and metrics in directories according to the 
# name given to each model

# New Facts, DHC and SC
python train.py -d /DATA/upal/Datasets/DHC/variations/new/var_4/cross_val/0_fold/not_dropped \
    -t /DATA/upal/Datasets/DHC/variations/new/var_4/targets/not_dropped/case_adv_winners.json \
    -n longformer_pred_winners \
    -p params_word2vec_200.json \
    -lm "allenai/longformer-base-4096" \
    -ul /DATA/upal/Datasets/DHC/variations/new/var_4/targets/not_dropped/selected_advs.txt \
    -id 0
