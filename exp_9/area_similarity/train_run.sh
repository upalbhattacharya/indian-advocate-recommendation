#!/usr/bin/env sh

python train.py -d /DATA/upal/Datasets/DHC/variations/new/var_4/cross_val/0_fold/with_areas \
    -t /DATA/upal/Datasets/DHC/variations/new/var_4/targets/with_areas/case_areas.json \
    -sbm "sentence-transformers/all-distilroberta-v1" \
    -n SBERT_distilroberta_fold_$fold \
    -p params_distilroberta.json 


