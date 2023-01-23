#!/usr/bin/sh

# Script to run several models one after another. All results will be placed in 
# experiments under model_states and metrics in directories according to the 
# name given to each model

# Top 6 
# ./train.py -e /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/bert_vanilla /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/bert_letsum /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/bert_kmm /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/inlegalbert_vanilla /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/inlegalbert_letsum /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/longformer \
#     	 -en bert_vanilla bert_letsum bert_kmm inlegalbert_vanilla inlegalbert_letsum longformer \
#        -ed 768 768 768 768 768 768 \
#	 -t /DATA/aniket/bert0_forward_modified/exp_4/case_advs.json \
#	 -ul /DATA/aniket/bert0_forward_modified/exp_4/adv_50.txt \
#	 -n adv_pred_ensemble \
#	 -p params_word2vec_200.json \
#	 -id 1

# All
./train.py -e /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/bert_vanilla /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/bert_letsum /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/bert_kmm /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/bert_casesummarizer /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/bert_summarunner_cnnrnn /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/bert_summarunner_rnnrnn /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/inlegalbert_vanilla /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/inlegalbert_letsum /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/inlegalbert_kmm /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/inlegalbert_casesummarizer /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/inlegalbert_summarunner_cnnrnn /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/inlegalbert_summarunner_rnnrnn /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/legalbert_vanilla /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/legalbert_letsum /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/legalbert_kmm /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/legalbert_casesummarizer /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/legalbert_summarunner_cnnrnn /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/legalbert_summarunner_rnnrnn /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/han /DATA/aniket/ensembles/embeddings/adv_pred_embeddings/longformer \
    	-en bert_vanilla bert_letsum bert_kmm bert_casesummarizer bert_summarunner_cnnrnn bert_summarunner_rnnrnn inlegalbert_vanilla inlegalbert_letsum inlegalbert_kmm inlegalbert_casesummarizer inlegalbert_summarunner_cnnrnn inlegalbert_summarunner_rnnrnn legalbert_vanilla legalbert_letsum legalbert_kmm legalbert_casesummarizer legalbert_summarunner_cnnrnn legalbert_summarunner_rnnrnn han longformer \
        -ed 768 768 768 768 768 768 768 768 768 768 768 768 768 768 768 768 768 768 200 768 \
	-t /DATA/aniket/bert0_forward_modified/exp_4/case_advs.json \
	-ul /DATA/aniket/bert0_forward_modified/exp_4/adv_50.txt \
        -n adv_pred_ensemble_all_embeds \
        -p params_word2vec_200.json \
	-id 1
