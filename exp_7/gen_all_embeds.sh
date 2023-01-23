#!/usr/bin/sh

FOLDS=4
echo "$(which python)"

for fold in $(seq 0 $FOLDS)
do
    python generate_embeddings.py -d ~/Results/advocate_recommendation/new/exp_3.1/cross_val/5_fold/fold_$fold/embeddings/train_rep \
    -t ~/Datasets/DHC/variations/new/var_1/targets/case_chapters.json \
    -p params_word2vec_200.json \
    -n doc2vec_embed_train_fold_$fold \
    -r ~/Repos/my_repos/advocate_recommendation/exp_7/experiments/model_states/doc2vec_embed_pred_fold_$fold/epoch_50.pth.tar \
    -s /home/workboots/Results/advocate_recommendation/new/exp_7/doc2vec/cross_val/5_fold/fold_$fold/embeddings/train 

    python generate_embeddings.py -d ~/Results/advocate_recommendation/new/exp_3.1/cross_val/5_fold/fold_$fold/embeddings/test_rep \
    -t ~/Datasets/DHC/variations/new/var_1/targets/case_chapters.json \
    -p params_word2vec_200.json \
    -n doc2vec_embed_test_fold_$fold \
    -r ~/Repos/my_repos/advocate_recommendation/exp_7/experiments/model_states/doc2vec_embed_pred_fold_$fold/epoch_50.pth.tar \
    -s /home/workboots/Results/advocate_recommendation/new/exp_7/doc2vec/cross_val/5_fold/fold_$fold/embeddings/test 
done

