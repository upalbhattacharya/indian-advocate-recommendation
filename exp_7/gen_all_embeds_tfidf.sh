#!/usr/bin/sh

fold=4

python generate_embeddings.py -d ~/Results/advocate_recommendation/new/exp_2.1/cross_val/5_fold/fold_$fold/embeddings/train_rep \
-t ~/Datasets/DHC/variations/new/var_1/targets/case_chapters.json \
-p params_word2vec_200_tf_idf.json \
-n tfidf_embed_train_fold_$fold \
-r ~/Repos/my_repos/advocate_recommendation/exp_7/experiments/model_states/tf_idf_embed_pred_fold_$fold/epoch_30.pth.tar \
-s /home/workboots/Results/advocate_recommendation/new/exp_7/tfidf/cross_val/5_fold/fold_$fold/embeddings/train 

python generate_embeddings.py -d ~/Results/advocate_recommendation/new/exp_2.1/cross_val/5_fold/fold_$fold/embeddings/test_rep \
-t ~/Datasets/DHC/variations/new/var_1/targets/case_chapters.json \
-p params_word2vec_200_tf_idf.json \
-n tfidf_embed_test_fold_$fold \
-r ~/Repos/my_repos/advocate_recommendation/exp_7/experiments/model_states/tf_idf_embed_pred_fold_$fold/epoch_30.pth.tar \
-s /home/workboots/Results/advocate_recommendation/new/exp_7/tfidf/cross_val/5_fold/fold_$fold/embeddings/test 

