#!/usr/bin/sh

# Script to run several models one after another. All results will be placed in 
# experiments under model_states and metrics in directories according to the 
# name given to each model


# Old Facts, only DHC
python train.py -d ~/Datasets/DHC/variations/var_1.1/data/ipc_data/cross_val/20_fold/fold_0 \
    -t ~/Datasets/DHC/variations/var_1.1/targets/ipc_case_offences.json \
    -n han_pred_DHC_old_facts -p params_word2vec_200.json


# New Facts, only DHC
python train.py -d ~/Datasets/DHC/variations/var_1.3/data/ipc_data/cross_val/20_fold/fold_0 \
    -t ~/Datasets/DHC/variations/var_1.3/targets/ipc_case_offences.json \
    -n han_pred_DHC_new_facts -p params_word2vec_200.json


# New Facts, DHC and SC
python train.py -d ~/Datasets/DHC/variations/var_1.3/data/ipc_data/cross_val/20_fold/fold_0/ ~/Datasets/SC/variations/var_1.1/data/ipc_data/fact_sentences \
    -t ~/Datasets/DHC/variations/var_1.3/targets/ipc_case_offences.json ~/Datasets/SC/variations/var_1.1/targets/ipc_case_offences.json \
    -n han_pred_DHC_SC_new_facts -p params_word2vec_200.json


# New Facts, DHC and SC, neg_ratio
python train.py -d ~/Datasets/DHC/variations/var_1.3/data/ipc_data/cross_val/20_fold/fold_0/ ~/Datasets/SC/variations/var_1.1/data/ipc_data/fact_sentences \
    -t ~/Datasets/DHC/variations/var_1.3/targets/ipc_case_offences.json ~/Datasets/SC/variations/var_1.1/targets/ipc_case_offences.json \
    -n han_pred_DHC_SC_new_facts -p params_word2vec_200_neg_ratio.json
