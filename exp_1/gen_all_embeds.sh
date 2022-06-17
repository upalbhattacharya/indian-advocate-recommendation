#!/usr/bin/sh

FOLDS=4
echo "$(which python)"

for fold in $(seq 0 $FOLDS)
do
    python generate_embeddings.py -d ~/Datasets/DHC/variations/var_3/data/ipc_data/cross_val/5_fold/fold_$fold/train \
    -t ~/Datasets/DHC/variations/var_3/targets/ipc_case_offences.json \
    -p params_word2vec_200.json \
    -r ~/Results/advocate_recommendation/exp_1_thresh_0.7/model_states/han_pred_DHC_SC_new_facts_fold_$fold/epoch_100.pth.tar \
    -n han_pred_DHC_SC_new_facts_gen_embeds_train_fold_$fold \
    -ul ~/Datasets/DHC/variations/var_3/targets/unique_labels.txt \
    -s ~/Results/advocate_recommendation/exp_1_thresh_0.7/embeddings/han_pred_DHC_SC_new_facts_fold_$fold/train

    python generate_embeddings.py -d ~/Datasets/DHC/variations/var_3/data/ipc_data/cross_val/5_fold/fold_$fold/test \
    -t ~/Datasets/DHC/variations/var_3/targets/ipc_case_offences.json \
    -p params_word2vec_200.json \
    -r ~/Results/advocate_recommendation/exp_1_thresh_0.7/model_states/han_pred_DHC_SC_new_facts_fold_$fold/epoch_100.pth.tar \
    -n han_pred_DHC_SC_new_facts_gen_embeds_test_fold_$fold \
    -ul ~/Datasets/DHC/variations/var_3/targets/unique_labels.txt \
    -s ~/Results/advocate_recommendation/exp_1_thresh_0.7/embeddings/han_pred_DHC_SC_new_facts_fold_$fold/test
done

