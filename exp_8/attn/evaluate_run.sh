#!/usr/bin/sh

# Script to run several models one after another. All results will be placed in 
# experiments under model_states and metrics in directories according to the 
# name given to each model

# Top 6
# ./evaluate.py -e /DATA/aniket/ensembles/embeddings/bert_vanilla/test /DATA/aniket/ensembles/embeddings/bert_letsum/test /DATA/aniket/ensembles/embeddings/bert_kmm/test /DATA/aniket/ensembles/embeddings/inlegalbert_vanilla/test /DATA/aniket/ensembles/embeddings/inlegalbert_letsum/test /DATA/aniket/ensembles/embeddings/longformer/test \
#     	 -en bert_vanilla bert_letsum bert_kmm inlegalbert_vanilla inlegalbert_letsum longformer \
#        -ed 768 768 768 768 768 768 \
#	 -t /DATA/aniket/bert0_forward_modified/exp_4/case_advs.json \
#	 -ul /DATA/aniket/bert0_forward_modified/exp_4/adv_50.txt \
#	 -n adv_pred_ensemble_test \
#	 -p params_word2vec_200.json \
#	 -r /DATA/aniket/ensembles/exp_6/attn/experiments/model_states/adv_pred_ensemble/best.pth.tar \
#	 -id 1

# All
./evaluate.py -e /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/bert_vanilla/test /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/bert_letsum/test /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/bert_kmm/test /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/bert_casesummarizer/test /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/bert_summarunner_cnnrnn/test /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/bert_summarunner_rnnrnn/test /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/inlegalbert_vanilla/test /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/inlegalbert_letsum/test /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/inlegalbert_kmm/test /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/inlegalbert_casesummarizer/test /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/inlegalbert_summarunner_cnnrnn/test /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/inlegalbert_summarunner_rnnrnn/test /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/legalbert_vanilla/test /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/legalbert_letsum/test /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/legalbert_kmm/test /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/legalbert_casesummarizer/test /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/legalbert_summarunner_cnnrnn/test /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/legalbert_summarunner_rnnrnn/test /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/han/test /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/longformer/test \
    	-en bert_vanilla bert_letsum bert_kmm bert_casesummarizer bert_summarunner_cnnrnn bert_summarunner_rnnrnn inlegalbert_vanilla inlegalbert_letsum inlegalbert_kmm inlegalbert_casesummarizer inlegalbert_summarunner_cnnrnn inlegalbert_summarunner_rnnrnn legalbert_vanilla legalbert_letsum legalbert_kmm legalbert_casesummarizer legalbert_summarunner_cnnrnn legalbert_summarunner_rnnrnn han longformer \
        -ed 768 768 768 768 768 768 768 768 768 768 768 768 768 768 768 768 768 768 200 768 \
	-t /DATA/aniket/bert0_forward_modified/exp_4/case_advs.json \
	-ul /DATA/aniket/bert0_forward_modified/exp_4/adv_50.txt \
        -n adv_pred_ensemble_all_embeds_test \
	-r /DATA/aniket/ensembles/exp_6/attn/experiments/model_states/adv_pred_ensemble_all_embeds/best.pth.tar \
        -p params_word2vec_200.json \
	-id 1
