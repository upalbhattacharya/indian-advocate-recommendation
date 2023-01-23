#!/usr/bin/sh

FOLDS=4

# Generate similarity ranking and re-ranking for all provided FOLDS

for fold in $(seq 0 $FOLDS)
do
    python calculate_ranks.py -d ~/Results/advocate_recommendation/exp_1_thresh_0.7/embeddings/han_pred_DHC_SC_new_facts_fold_$fold/train/ \
        -q ~/Results/advocate_recommendation/exp_1_thresh_0.7/embeddings/han_pred_DHC_SC_new_facts_fold_$fold/test/ \
        -ct ~/Datasets/DHC/common/case_advs.json \
        -a ~/Datasets/DHC/common/selected_advs.json \
        -o ~/Results/advocate_recommendation/exp_1_thresh_0.7/metrics/han_pred_DHC_SC_new_facts_fold_$fold/
done
