#!/usr/bin/sh

FOLDS=4
echo "$(which python)"

for fold in $(seq 0 $FOLDS)
do
    python generate_embeddings.py -d ~/Datasets/DHC/variations/var_3/data/ipc_data/adv_rec_eval_splits/fold_$fold/train \
    -t ~/Datasets/DHC/variations/var_3/targets/ipc_case_offences.json \
    -p params_distilroberta.json \
    -n sbert_distilroberta_train_fold_$fold \
    -ul ~/Datasets/DHC/variations/var_3/targets/unique_labels.txt \
    -s ~/Results/advocate_recommendation/exp_5/no_fine_tune/embeddings/sbert_distilroberta_fold_$fold/train

    python generate_embeddings.py -d ~/Datasets/DHC/variations/var_3/data/ipc_data/adv_rec_eval_splits/fold_$fold/test \
    -t ~/Datasets/DHC/variations/var_3/targets/ipc_case_offences.json \
    -p params_distilroberta.json \
    -n sbert_distilroberta_test_fold_$fold \
    -ul ~/Datasets/DHC/variations/var_3/targets/unique_labels.txt \
    -s ~/Results/advocate_recommendation/exp_5/no_fine_tune/embeddings/sbert_distilroberta_fold_$fold/test

done
