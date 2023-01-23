#!/usr/bin/env zsh

FOLDS=4
THRESHOLDS=(
    1.0
    0.80
)

for i in $(seq 0 $FOLDS)
do
    ./map.py \
    -d ~/Results/advocate_recommendation/new/exp_2.1/cross_val/5_fold/fold_$i/results/similarity_reranking.json \
    -i ~/Datasets/DHC/variations/new/var_1/cross_val/5_fold/fold_$i/adv_split_info.json \
    -t ~/Datasets/DHC/variations/new/var_1/targets/case_advs.json \
    -o ~/Results/advocate_recommendation/new/exp_2.1/cross_val/5_fold/fold_$i/metrics/ \
    -l ~/Results/advocate_recommendation/new/exp_2.1/cross_val/5_fold/fold_$i/logs/ \
    -n map_hard
done

for i in $(seq 0 $FOLDS)
do
    ./map.py \
    -d ~/Results/advocate_recommendation/new/exp_2.1/cross_val/5_fold/fold_$i/results/similarity_reranking.json \
    -i ~/Datasets/DHC/variations/new/var_1/cross_val/5_fold/fold_$i/adv_split_info.json \
    -t ~/Datasets/DHC/variations/new/var_1/targets/case_winners.json \
    -o ~/Results/advocate_recommendation/new/exp_2.1/cross_val/5_fold/fold_$i/metrics/ \
    -l ~/Results/advocate_recommendation/new/exp_2.1/cross_val/5_fold/fold_$i/logs/ \
    -n map_harder
done


# for score in ${THRESHOLDS[@]}
# do
    # for i in $(seq 0 $FOLDS)
    # do
        # ./map.py \
        # -d ~/Results/advocate_recommendation/exp_2/new_cross_val/20_fold/fold_$i/scores.json \
        # -i ~/Datasets/DHC/variations/var_2/data/ipc_data/cross_val/20_fold/fold_$i/adv_case_splits.json \
        # -t ~/Datasets/DHC/common/case_advs.json \
        # -n map_$score \
        # -c ~/Datasets/DHC/variations/var_2/targets/ipc_case_offences.json \
        # -ac ~/Datasets/DHC/variations/var_2/targets/adv_ipc_charges.json \
        # -th $score
    # done
# done
